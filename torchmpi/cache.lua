local MPI = require("torchmpi.env")

local cache = {}

-- Depending on the type of memory allocated on either the CPU or GPU, torch_mpi
-- clones and memoizes tensors. The rules are:
-- 1. CPU tensors are never copied or converted
-- 2. Cuda tensors backed by UVA memory are converted back and forth to Float
--    tensors by passing the raw pointer. According to pre-6.x compute capability
--    UVA coherence rules, we perform synchronization with stream 0 around the MPI
--    calls.
-- 3. Cuda tensors not backed by UVA memory are copied back and forth to pinned
--    CPU memory, collectives are run on this (TODO: async copies etc)
-- 4. Future: CudaIpc memory
-- 5. Future: RDMA
-- As a consequence, we provide an API to free the references to the copies we
-- maintain for collectives.
-- After calling those free functions, you should consider whether you want to
-- call collectgarbage twice.
cache.tensorCache = {}
cache.tensorReferences = {}
cache.parameterServers = {}
cache.extraTensorReferences = {}
cache.prefetchTensorReferences = {}
cache.momentumTensorReferences = {}
cache.freeReferencesToTensor = function(t)
   for k, v in pairs(cache.tensorCache) do
      if k == t or v == t then
         cache.tensorCache[k] = nil
      end
   end
   for k, v in pairs(cache.tensorReferences) do
      if k == t or v.orig == t or v.converted == t then
         cache.tensorReferences[k] = nil
      end
   end
   for k, v in pairs(cache.prefetchTensorReferences) do
      if k == t or v.orig == t or v.converted == t then
         cache.prefetchTensorReferences[k] = nil
      end
   end
   for k, v in pairs(cache.extraTensorReferences) do
      if k == t or v.orig == t or v.converted == t then
         cache.extraTensorReferences[k] = nil
      end
   end
   for k, v in pairs(cache.momentumTensorReferences) do
      if k == t or v.orig == t or v.converted == t then
         cache.momentumTensorReferences[k] = nil
      end
   end
   for k, ps in pairs(cache.parameterServers) do
      if k == t or v.orig == t or v.converted == t then
         cache.parameterserverfree(ps)
         cache.parameterServers[k] = nil
      end
   end
end

cache.freeAllTensorReferences = function()
   cache.tensorCache = {}
   cache.tensorReferences = {}
   for t, ps in pairs(cache.parameterServers) do
      cache.parameterserverfree(ps)
   end
   cache.parameterServers = {}
   cache.extraTensorReferences = {}
   cache.momentumTensorReferences = {}
   cache.prefetchTensorReferences = {}
end

cache.convert = function(t, nowarning)
   assert(torch.isTensor(t), 'Not a tensor ' .. torch.type(t))
   if cache.tensorReferences[t] then
      assert(cache.tensorReferences[t].orig == t)
      if cache.tensorReferences[t].needsCopy then
         cache.tensorReferences[t].converted:copy(t)
      end
      return cache.tensorReferences[t].converted, cache.tensorReferences[t].orig
   end
   cache.tensorReferences[t] = {}
   cache.tensorReferences[t].orig = t
   if torch.type(t):find('Cuda') then
      if MPI.supportsCuda then
         cache.tensorReferences[t].converted = t
         cache.tensorReferences[t].needsCopy = false
      else
         if not nowarning then
            print('[MPI] Warning: copying to pinned CPU memory on the fly for collective, '..
                     'make sure you memoize the tensor')
         end
         cache.tensorReferences[t].converted = cache.toCPUPinned(t)
         cache.tensorReferences[t].needsCopy = true -- for unconvert
         cache.tensorReferences[t].converted:copyAsync(t)
      end
   else
      cache.tensorReferences[t].converted = t
      cache.tensorReferences[t].needsCopy = false
   end
   return cache.tensorReferences[t].converted, cache.tensorReferences[t].orig
end

cache.unconvert = function(t, orig)
   assert(cache.tensorReferences[orig].converted == t)
   assert(cache.tensorReferences[orig].orig == orig)
   -- copy back
   if cache.tensorReferences[orig].needsCopy then orig:copy(t) end
   return orig
end

-- These function recursively copy *THTensors using the proper THCudaTensor
-- equivalent. Unfortunately some of torch.nn modules don't work with this
-- (e.g. ClassNLLCriterion works with IntTensor on CPU and CudaTensor
--    (a.k.a CudaFloatTensor) on GPU.
-- Use sparingly atm ... until we wipe all this
cache.toGPU = function(t)
   assert(withCuda)
   if not torch.isTensor(t) then
      -- Not a Tensor
      if type(t) ~= 'table' then return t end
      for k, v in pairs(t) do
         if torch.type(t):match('ClassNLLCrit') then
            print('toGPU ', k, 'of type', torch.type(v))
         end
         t[k] = cache.toGPU(v)
         if torch.type(t):match('ClassNLLCrit') then
            print('Now ', k, 'of type', torch.type(t[k]))
         end
      end
      return t
   end
   -- Tensor

   if torch.type(t):match('Cuda') then return t end
   -- Not a CudaTensor

   -- Already converted and shared
   if cache.tensorCache[t] then return cache.tensorCache[t] end

   local typ = torch.type(t):gsub('torch.', 'Cuda')
   typ = (typ == 'CudaFloatTensor') and 'CudaTensor' or typ -- sigh
   local res = torch[typ]()
   res:resize(t:size())
   res:copy(t)

   cache.tensorCache[t] = res
   return res
end

_toCPU = function(t, pinned)
   if not torch.isTensor(t) then
      -- Not a Tensor
      if not type(t) ~= 'table' then return t end
      for k, v in pairs(t) do t[k] = _toCPU(v, pinned) end
      return t
   end
   -- Tensor

   -- If not a CudaTensor, return early
   if not torch.type(t):match('Cuda') then return t end

   -- CudaXYZTensor
   -- Already converted and shared
   if cache.tensorCache[t] then return cache.tensorCache[t] end

   local typ = torch.type(t):gsub('torch.Cuda', '')
   -- sigh some modules in NN convert anything and the kitchen sink to
   -- CudaTensor and cutorch only support CudaTensors as pinned
   typ = (typ == 'Tensor') and 'FloatTensor' or typ
   if typ == 'FloatTensor' and pinned then
      local res = cutorch.createCudaHostTensor(t:size())
      res:copy(t)
      cache.tensorCache[t] = res
   else
      local res = torch[typ]()
      res:resize(t:size())
      res:copy(t)
      cache.tensorCache[t] = res
   end
   return cache.tensorCache[t]
end

cache.toCPU = function(t) return _toCPU(t) end
cache.toCPUPinned = function(t) return _toCPU(t, true) end

require('ffi')
cache.freeDescriptors = function()
   MPI.C.torchmpi_free_ipc_descriptors()
end

return cache
