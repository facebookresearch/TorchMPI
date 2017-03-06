require('torch')

local cmd = torch.CmdLine()
cmd:option('-inPlace', false, 'run inPlace or not')
cmd:option('-gpu', false, 'run on gpu (must be true)')
cmd:option('-async', false, 'dispatch collectives asynchronously and wait for handle before checking')
cmd:option('-benchmark', false, 'skip correctness check and benchmark performance instead')
cmd:option('-all', false, 'run all configs')

local config = cmd:parse(arg)
config.check = not config.benchmark
assert(config.gpu, 'This test must be ran with GPUs, please specify -gpu.')

local nSkip = config.benchmark and 10 or 0
local nRuns = config.benchmark and 10 + nSkip or 1

-- If using GPUs, set the GPU before initializing MPI
local mpi = require('torchmpi')
local tester = require('torchmpi.tester')
mpi.start(config.gpu, false)

local tests = {}

local asyncTimer = torch.Timer()

-------------------------------- reduce --------------------------------
tests.reduce = {}
tests.reduce.test = function(input, output, firstRun)
   local ns = config.async and mpi.async.nccl or mpi.nccl

   if config.async then
      asyncTimer = torch.Timer()
   end

   local ok, res = pcall(ns.reduceTensor, 0, input, output)

   if not ok then
      assert(res:find('NYI:'))
      return 'NYI'
   end

   if config.async and not firstRun then
      asyncTimer:stop()
      if asyncTimer:time().real >= 5e-5 then
         print(string.format(
            'Warning: Async broadcast launch took %f (expected < %f) for size %d',
            asyncTimer:time().real, 5e-5, input:nElement()))
      end
   end

   return res
end

-- Assumes a pipelined implementation of reduce
tests.reduce.communicationVolumeGB = function(input)
   local elemSize = 4
   return (input:nElement() * elemSize) / 1e9
end

-- Careful, precision counts here once we reach a certain size
tests.reduce.check = function(input, output)
   local tocheck = output
   -- 0-based
   local val = (mpi.size() * (mpi.size() - 1)) / 2
   if mpi.rank() == 0 then
      assert(tocheck:min() == val,
         tostring(tocheck:min()) .. ' VS expected ' .. tostring(val))
      assert(tocheck:max() == val,
         tostring(tocheck:max()) .. ' VS expected ' .. tostring(val))
   end
   mpi.barrier()
end


-------------------------------- allreduce --------------------------------
tests.allreduce = {}
tests.allreduce.test = function(input, output, firstRun)
   local ns = config.async and mpi.async.nccl or mpi.nccl

   if config.async then
      asyncTimer = torch.Timer()
   end

   local res = ns.allreduceTensor(input, output)

   if config.async and not firstRun then
      asyncTimer:stop()
      if asyncTimer:time().real >= 5e-5 then
         print(string.format(
            'Warning: Async broadcast launch took %f (expected < %f) for size %d',
            asyncTimer:time().real, 5e-5, input:nElement()))
      end
   end

   return res
end

-- Assumes a chunked-ring-based implementation of allreduce
-- (i.e. 1 roundtrip of the whole data through slowest wire to saturate BW)
tests.allreduce.communicationVolumeGB = function(input)
   local elemSize = 4
   return (2 * input:nElement() * elemSize * (mpi.size() - 1) / mpi.size()) / 1e9
end

tests.allreduce.check = function(input, output)
   local tocheck = output
   -- 0-based
   local val = (mpi.size() * (mpi.size() - 1)) / 2
   local min, max = tocheck:min(), tocheck:max()
   -- Careful, precision counts here once we reach a certain size
   assert(min == val, tostring(min) .. ' VS expected ' .. tostring(val))
   assert(max == val, tostring(max) .. ' VS expected ' .. tostring(val))
end


-------------------------------- broadcast --------------------------------
tests.broadcast = {}
tests.broadcast.test = function(input, output, firstRun)
   local ns = config.async and mpi.async.nccl or mpi.nccl

   if config.async then
      asyncTimer = torch.Timer()
   end

   local ok, res = pcall(ns.broadcastTensor, mpi.size() - 1, input)

   if not ok then
      assert(res:find('NYI:'))
      return 'NYI'
   end

   if config.async and not firstRun then
      asyncTimer:stop()
      if asyncTimer:time().real >= 5e-5 then
         print(string.format(
            'Warning: Async broadcast launch took %f (expected < %f) for size %d',
            asyncTimer:time().real, 5e-5, input:nElement()))
      end
   end

   return res
end

-- Assumes a pipelined implementation of broadcast
tests.broadcast.communicationVolumeGB = function(input)
   local elemSize = 4
   return (input:nElement() * elemSize) / 1e9
end

-- Careful, precision counts here once we reach a certain size
tests.broadcast.check = function(input)
   -- 0-based
   local val = mpi.size() - 1
   local min, max = input:min(), input:max()
   -- Careful, precision counts here once we reach a certain size
   assert(min == val, tostring(min) .. ' VS expected ' .. tostring(val))
   assert(max == val, tostring(max) .. ' VS expected ' .. tostring(val))
end

-------------------------------- Start tests --------------------------------
tests.broadcast.implemented = mpi.hasNCCL
tests.reduce.implemented = mpi.hasNCCL
tests.allreduce.implemented = mpi.hasNCCL

if config.all then
   for _, async in ipairs({false, true}) do
      for _, inPlace in ipairs({false, true}) do
         config.async, config.inPlace =
            async or config.async,
           inPlace or config.inPlace
         tester.runOneConfig(tests, nRuns, nSkip, config)
      end
   end
else
   tester.runOneConfig(tests, nRuns, nSkip, config)
end

mpi.barrier()
if mpi.rank() == 0 then print('Success') end
mpi.stop()
