--[[
 Copyright (c) 2016-present, Facebook, Inc.
 All rights reserved.

 This source code is licensed under the BSD-style license found in the
 LICENSE file in the root directory of this source tree. An additional grant
 of patent rights can be found in the PATENTS file in the same directory.
--]]
require('torch')

local cmd = torch.CmdLine()
cmd:option('-benchmark', false, 'skip correctness check and benchmark performance instead')
cmd:option('-tests', 'all', 'Options: all | allselector | basic | p2p | nccl')
cmd:option('-processor', 'both', 'Options: gpu | cpu | both')
cmd:option('-async', false, 'dispatch collectives asynchronously and wait for handle before checking')
cmd:option('-inPlace', false, 'run inPlace or not')

local config = cmd:parse(arg)
assert(config.tests == 'all' or config.tests == 'allselector' or
       config.tests == 'basic' or config.tests == 'p2p' or config.tests == 'nccl')

local gpuTable = nil
if config.processor == 'gpu' then
   gpuTable = {true}
elseif config.processor == 'cpu' then
   gpuTable = {false}
elseif config.processor == 'both' then
   gpuTable = {false, true}
else
   error("Illegal processor option %s", config.processor)
end

-- these settings are a little strange but somewhat preserves old behavior,
-- i.e. async=true will only run async tests, but async=false will run
-- both false and true
local asyncTable = config.async and {false, true} or {true}
local inPlaceTable = config.inPlace and {false, true} or {true}

config.check = not config.benchmark
if config.tests == 'nccl' then
   assert(config.processor == "gpu", 'This test must be ran with GPUs, please specify -gpu.')
end

local nSkip = config.benchmark and 10 or 0
local nRuns = config.benchmark and 10 + nSkip or 1

-- If using GPUs, set the GPU before initializing MPI
local mpi = require('torchmpi')
mpi.start(true)

if false then
   -- mpi.C.torchmpi_set_flat_collectives()
   mpi.C.torchmpi_set_hierarchical_collectives()
   -- mpi.C.torchmpi_set_staged_collectives()
   mpi.C.torchmpi_set_direct_collectives()
   mpi.C.torchmpi_set_num_buffers_per_cpu_collective(4)
   mpi.C.torchmpi_set_num_buffers_per_gpu_collective(4)
   mpi.C.torchmpi_set_min_buffer_size_per_cpu_collective(2^10)
   mpi.C.torchmpi_set_max_buffer_size_per_cpu_collective(2^20)
   mpi.C.torchmpi_set_min_buffer_size_per_gpu_collective(2^10)
   mpi.C.torchmpi_set_max_buffer_size_per_gpu_collective(2^20)
   mpi.C.torchmpi_set_broadcast_size_cpu_tree_based(2^22)
   mpi.C.torchmpi_set_broadcast_size_gpu_tree_based(2^22)
end

local collectiveAvailabilityCPU = mpi.collectiveAvailability(true, false)
local collectiveAvailabilityGPU = mpi.collectiveAvailability(false, true)
local tester = require('torchmpi.tester')
local asyncTimer = torch.Timer()

local function getCollectives()
   if config.tests == "allselector" then
      local sel = mpi.collectiveSelector
      sel = config.gpu and sel.gpu or sel.cpu
      sel = config.singlenode and sel.singlenode or sel.multinode
      return config.async and sel.async or sel.async
   else
      local sel = mpi
      if config.nccl then
         sel = mpi.hasNCCL and sel.nccl or nil
      end
      if sel ~= nil then
        sel = config.async and sel.async or sel
        sel = config.p2p and sel.p2p or sel
      end

      return sel
   end
end

local function collectiveAvailable(ns, collective)
   if config.tests == "all" then
      local funcname = "MPI" .. (config.nccl and ".nccl" or "" )
         .. (config.async and ".async" or "") .. (config.p2p and ".p2p." or ".")
         .. collective
      local availability = config.gpu and
         collectiveAvailabilityGPU or collectiveAvailabilityCPU
      local func = availability:match(funcname .. '[^\n]+')
      if func:match('unavailable') then
         return 'UNAVAILABLE'
      elseif func:match('unimplemented') then
         return 'NYI'
      end
   end
   return 'available'
end

local tests = {}

-------------------------------- reduce --------------------------------
tests.reduce = {}
tests.reduce.test = function(input, output, firstRun)
   -- Output must be zeroed explicitly to get proper results,
   -- only when out of place
   if input ~= output then output:zero() end

   local ns = getCollectives()
   local availability = collectiveAvailable(ns, "reduceTensor")
   if availability ~= 'available' then
      return availability -- exit test loop
   end

   if config.async then
      asyncTimer:reset()
   end

   local handle = ns.reduceTensor(0, input, output)

   if config.async and not firstRun then
      asyncTimer:stop()
      if asyncTimer:time().real >= 5e-5 then
         print(string.format(
            'Warning: Async reduce launch took %f (expected < %f) for size %d',
            asyncTimer:time().real, 5e-5, input:nElement()))
      end
   end

   return handle
end

-- Careful, precision counts here once we reach a certain size
tests.reduce.check = function(input, output)
   if mpi.rank() == 0 then
      local val = (mpi.size() * (mpi.size() - 1)) / 2
      local min, max = output:min(), output:max()
      assert(min == val, ('%f vs expected %f'):format(min, val))
      assert(max == val, ('%f vs expected %f'):format(max, val))
   end
end

-- Assumes a pipelined implementation of reduce
tests.reduce.communicationVolumeGB = function(input)
   local elemSize = 4
   return (input:nElement() * elemSize) / 1e9
end

-------------------------------- broadcast --------------------------------
tests.broadcast = {}

tests.broadcast.test = function(input, output, firstRun)
   local ns = getCollectives()
   local availability = collectiveAvailable(ns, "broadcastTensor")
   if availability ~= 'available' then
      return availability -- exit test loop
   end

   if config.async then
      asyncTimer:reset()
   end

   local handle = ns.broadcastTensor(mpi.size() - 1, input)

   if config.async and not firstRun then
      asyncTimer:stop()
      if asyncTimer:time().real >= 5e-5 then
         print(string.format(
            'Warning: Async broadcast launch took %f (expected < %f) for size %d',
            asyncTimer:time().real, 5e-5, input:nElement()))
      end
   end

   return handle
end

-- Careful, precision counts here once we reach a certain size
tests.broadcast.check = function(input)
   -- only 1 tensor, no input/output distinction
   -- 0-based
   local val = mpi.size() - 1
   local min, max = input:min(), input:max()
   if min ~= val or max ~= val then
      error(('[%d/%d] %f vs expected %f %s (size: %d)\n'):format(
            mpi.rank(), mpi.size(), min, val, input:data(), input:nElement()))
   end
end

-- Assumes a pipelined implementation of broadcast
tests.broadcast.communicationVolumeGB = function(input)
   local elemSize = 4
   return (input:nElement() * elemSize) / 1e9
end

-------------------------------- allreduce --------------------------------
tests.allreduce = {}

tests.allreduce.test = function(input, output, firstRun)
   -- Output must be zeroed explicitly to get proper results, only when out of place
   if input ~= output then output:zero() end

   local ns = getCollectives()
   local availability = collectiveAvailable(ns, "allreduceTensor")
   if availability ~= 'available' then
      return availability -- exit test loop
   end

   if config.async then
      asyncTimer:reset()
   end

   local handle = ns.allreduceTensor(input, output)

   if config.async and not firstRun then
      asyncTimer:stop()
      if asyncTimer:time().real >= 5e-5 then
         print(string.format(
            'Warning: Async allreduce launch took %f (expected < %f) for size %d',
            asyncTimer:time().real, 5e-5, input:nElement()))
      end
   end

   return handle
end

-- Careful, precision counts here once we reach a certain size
tests.allreduce.check = function(input, output)
   local val = (mpi.size() * (mpi.size() - 1)) / 2
   local min, max = output:min(), output:max()
   if min ~= val or max ~= val then
      error(('[%d/%d] %f-%f vs expected %f (size %d)\n'):format(
            mpi:rank(), mpi:size(), min, max, val, output:nElement()))
   end
end

-- Assumes a chunked-ring-based implementation of allreduce
-- (i.e. 1 roundtrip of the whole data through slowest wire to saturate BW)
tests.allreduce.communicationVolumeGB = function(input)
   local elemSize = 4
   return (2 * input:nElement() * elemSize * (mpi.size() - 1) / mpi.size()) / 1e9
end

-------------------------------- sendreceivenext -------------------------------
tests.sendreceivenext = {}

local dist = 1
tests.sendreceivenext.dist = math.min(dist, mpi.size() - 1)
tests.sendreceivenext.test = function(input, output, firstRun)
   local ns = getCollectives()
   local availability = collectiveAvailable(ns, "sendreceiveTensor")
   if availability ~= 'available' then
      return availability -- exit test loop
   end

   if config.async then
      asyncTimer:reset()
   end

   local handle = ns.sendreceiveTensor(
      input,
      (mpi.rank() - tests.sendreceivenext.dist) % mpi.size(),
      (mpi.rank() + tests.sendreceivenext.dist) % mpi.size())

   if config.async and not firstRun then
      asyncTimer:stop()
      if asyncTimer:time().real >= 5e-5 then
         print(string.format(
            'Warning: Async sendreceivenext launch took %f (expected < %f) for size %d',
            asyncTimer:time().real, 5e-5, input:nElement())
         )
      end
   end

   return handle
end

-- Careful, precision counts here once we reach a certain size
tests.sendreceivenext.check = function(input, output)
   output:copy(input)
   local val = (mpi.rank() - tests.sendreceivenext.dist) % mpi.size()
   local min, max = output:min(), output:max()
   assert(min == val)
   assert(max == val)
end

-- Pure point-to-point, 1 hop
tests.sendreceivenext.communicationVolumeGB = function(input)
   local elemSize = 4
   return input:nElement() * elemSize / 1e9
end

-------------------------------- barrier --------------------------------
tests.mpiBarrier = {}
tests.mpiBarrier.test = function()
   mpi.barrier()
end
tests.mpiBarrier.check = function() end
tests.mpiBarrier.communicationVolumeGB = function() return 0 end

-------------------------------- custom barrier --------------------------------
local ffi = require('ffi')
ffi.cdef [[
   void customBarrier();
]]
tests.customBarrier = {}
tests.customBarrier.test = function()
   ffi.C.customBarrier()
end
tests.customBarrier.check = function() end
tests.customBarrier.communicationVolumeGB = function() return 0 end

-------------------------------- Start tests --------------------------------\
local function setImplemented()
   if config.tests == "all" or config.tests == "allselector" then
      tests.broadcast.implemented = true
      tests.reduce.implemented = true
      tests.allreduce.implemented = true
      tests.sendreceivenext.implemented = true
   elseif config.tests == "basic" then
      -- No async sendreceivenext
      tests.sendreceivenext.implemented = not config.async
      tests.mpiBarrier.implemented = true
      -- Disable because it deadlocks on multi-machines
      tests.customBarrier.implemented = false
      tests.broadcast.implemented = true
   -- No async sendreceivenextGPU reduce
      tests.reduce.implemented = not (config.async and config.gpu)
      tests.allreduce.implemented = true
   elseif config.tests == "p2p" then
      tests.broadcast.implemented = true
      tests.allreduce.implemented = true
   elseif config.tests == "nccl" then
      tests.broadcast.implemented = mpi.hasNCCL
      tests.reduce.implemented = mpi.hasNCCL
      tests.allreduce.implemented = mpi.hasNCCL
   end
end

local function ncclTable(gpu)
   if config.tests == "all" and gpu then
      return {false, true}
   elseif config.tests == "all" and not gpu then
      return {false}
   elseif config.tests == "nccl" then
      return {true}
   else
      return {false}
   end
end

if config.tests == "allselector" then
   for _, async in ipairs(asyncTable) do
      for _, gpu in ipairs(gpuTable) do
         for _, inPlace in ipairs(inPlaceTable) do
            for _, singlenode in ipairs({false, true}) do
               config.async = async
               config.gpu = gpu
               config.inPlace = inPlace
               config.singlenode = singlenode
               setImplemented()
               tester.runOneConfig(tests, nRuns, nSkip, config)
            end
         end
      end
   end
else
   for _, async in ipairs(asyncTable) do
      for _, gpu in ipairs(gpuTable) do
         for _, inPlace in ipairs(inPlaceTable) do
            local p2pTable = config.tests == "all" and {false, true}
                             or config.tests == "p2p" and {true} or {false}
            for _, p2p in ipairs(p2pTable) do
               for _, nccl in ipairs(ncclTable(gpu)) do
                  config.async = async
                  config.gpu = gpu
                  config.nccl = nccl
                  config.p2p = p2p
                  setImplemented()
                  tester.runOneConfig(tests, nRuns, nSkip, config)
                end
            end
         end
      end
   end
end

mpi.barrier()
if mpi.rank() == 0 then print('Success') end
mpi.stop()
