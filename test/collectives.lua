--[[
 Copyright (c) 2016-present, Facebook, Inc.
 All rights reserved.

 This source code is licensed under the BSD-style license found in the
 LICENSE file in the root directory of this source tree. An additional grant
 of patent rights can be found in the PATENTS file in the same directory.
--]]
require('torch')

local cmd = torch.CmdLine()
cmd:option('-inPlace', false, 'run inPlace or not')
cmd:option('-gpu', false, 'run on gpu')
cmd:option('-async', false, 'dispatch collectives asynchronously and wait for handle before checking')
cmd:option('-benchmark', false, 'skip correctness check and benchmark performance instead')
cmd:option('-all', false, 'run all configs')

local config = cmd:parse(arg)
config.check = not config.benchmark

local nSkip = config.benchmark and 10 or 0
local nRuns = config.benchmark and 10 + nSkip or 1

local initGPU = config.gpu
if not initGPU then
   cutorch = nil -- global name declaration for terra
end

local mpi = require('torchmpi')
local tester = require('torchmpi.tester')
mpi.start(initGPU)

local tests = {}

local asyncTimer = torch.Timer()

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

-------------------------------- broadcast --------------------------------
tests.broadcast = {}

tests.broadcast.test = function(input, output, firstRun)
   -- mpi or mpi.async
   local ns = config.async and mpi.async or mpi

   if config.async then
      asyncTimer = torch.Timer()
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
tests.broadcast.check = function(input) -- only 1 tensor, no input/output distinction
   -- 0-based
   local val = mpi.size() - 1
   local min, max = input:min(), input:max()
   assert(min == val, ('%f vs expected %f'):format(min, val) .. tostring(input))
   assert(max == val)
end

-- Assumes a pipelined implementation of broadcast
tests.broadcast.communicationVolumeGB = function(t)
   local elemSize = 4
   return (t:nElement() * elemSize) / 1e9
end


-------------------------------- reduce --------------------------------
tests.reduce = {}

tests.reduce.test = function(input, output, firstRun)
   -- Output must be zeroed explicitly to get proper results, only when out of place
   if input ~= output then output:zero() end

   -- mpi or mpi.async
   local ns = config.async and mpi.async or mpi

   if config.async then
      asyncTimer = torch.Timer()
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

-- Assumes a pipelined implementation of broadcast
tests.reduce.communicationVolumeGB = function(t)
   local elemSize = 4
   return (t:nElement() * elemSize) / 1e9
end

-------------------------------- allreduce --------------------------------
tests.allreduce = {}

tests.allreduce.test = function(input, output, firstRun)
   -- Output must be zeroed explicitly to get proper results, only when out of place
   if input ~= output then output:zero() end

   -- mpi or mpi.async
   local ns = config.async and mpi.async or mpi

   if config.async then
      asyncTimer = torch.Timer()
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
   assert(min == val, ('%f vs expected %f'):format(min, val))
   assert(max == val, ('%f vs expected %f'):format(max, val))
end

-- Assumes a tree-based implementation of allreduce
tests.allreduce.communicationVolumeGB = function(input)
   local elemSize = 4
   return (2 * input:nElement() * elemSize * (mpi.size() - 1) / mpi.size()) / 1e9
end


-------------------------------- sendreceivenext --------------------------------
tests.sendreceivenext = {}

local dist = 1
tests.sendreceivenext.dist = math.min(dist, mpi.size() - 1)
tests.sendreceivenext.test = function(input, output, firstRun)
   -- mpi or mpi.async
   local ns = config.async and mpi.async or mpi

   if config.async then
      asyncTimer = torch.Timer()
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
            asyncTimer:time().real, 5e-5, input:nElement()))
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

-------------------------------- Start tests --------------------------------

local function setImplemented()
  -- No async sendreceivenext
  tests.sendreceivenext.implemented = not config.async
  tests.mpiBarrier.implemented = true
  -- Disable because it deadlocks on multi-machines
  tests.customBarrier.implemented = false
  tests.broadcast.implemented = true
  -- No async sendreceivenextGPU reduce
  tests.reduce.implemented = not (config.async and config.gpu)
  tests.allreduce.implemented = true
end

if config.all then
   for _, async in ipairs({false, true}) do
      for _, inPlace in ipairs({false, true}) do
         config.async, config.inPlace =
            config.async or async, config.inPlace or inPlace
         setImplemented()
         tester.runOneConfig(tests, nRuns, nSkip, config)
      end
   end
else
   setImplemented()
   tester.runOneConfig(tests, nRuns, nSkip, config)
end

if mpi.rank() == 0 then print('Success') end

mpi.barrier()
mpi.stop()
