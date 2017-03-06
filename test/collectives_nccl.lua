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
tests.reduce.test = function(t, output)
   local ns = config.async and mpi.async.nccl or mpi.nccl

   if config.async then
      asyncTimer = torch.Timer()
   end

   local ok, res = pcall(ns.reduceTensor, 0, t, output)

   if not ok then
      assert(res:find('NYI:'))
      return 'NYI'
   end

   if config.async then
      asyncTimer:stop()
      if asyncTimer:time().real >= 5e-5 then
         print(string.format(
            'Warning: Async broadcast launch took %f (expected < %f) for size %d',
            asyncTimer:time().real, 5e-5, t:nElement()))
      end
   end

   return res
end

-- Assumes a pipelined implementation of reduce
tests.reduce.communicationVolumeGB = function(t)
   local elemSize = 4
   return (t:nElement() * elemSize) / 1e9
end

-- Careful, precision counts here once we reach a certain size
tests.reduce.check = function(t, output)
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
tests.allreduce.test = function(t, output)
   local ns = config.async and mpi.async.nccl or mpi.nccl

   if config.async then
      asyncTimer = torch.Timer()
   end

   local res = ns.allreduceTensor(t, output)

   if config.async then
      asyncTimer:stop()
      if asyncTimer:time().real >= 5e-5 then
         print(string.format(
            'Warning: Async broadcast launch took %f (expected < %f) for size %d',
            asyncTimer:time().real, 5e-5, t:nElement()))
      end
   end

   return res
end

-- Assumes a chunked-ring-based implementation of allreduce
-- (i.e. 1 roundtrip of the whole data through slowest wire to saturate BW)
tests.allreduce.communicationVolumeGB = function(t)
   local elemSize = 4
   return (2 * t:nElement() * elemSize * (mpi.size() - 1) / mpi.size()) / 1e9
end

tests.allreduce.check = function(t, output)
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
tests.broadcast.test = function(t)
   local ns = config.async and mpi.async.nccl or mpi.nccl

   if config.async then
      asyncTimer = torch.Timer()
   end

   local ok, res = pcall(ns.broadcastTensor, mpi.size() - 1, t)

   if not ok then
      assert(res:find('NYI:'))
      return 'NYI'
   end

   if config.async then
      asyncTimer:stop()
      if asyncTimer:time().real >= 5e-5 then
         print(string.format(
            'Warning: Async broadcast launch took %f (expected < %f) for size %d',
            asyncTimer:time().real, 5e-5, t:nElement()))
      end
   end

   return res
end

-- Assumes a pipelined implementation of broadcast
tests.broadcast.communicationVolumeGB = function(t)
   local elemSize = 4
   return (t:nElement() * elemSize) / 1e9
end

-- Careful, precision counts here once we reach a certain size
tests.broadcast.check = function(t)
   -- 0-based
   local val = mpi.size() - 1
   local min, max = t:min(), t:max()
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
