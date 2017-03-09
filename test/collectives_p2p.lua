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

local nSkip = config.benchmark and 10 or 0
local nRuns = config.benchmark and 10 + nSkip or 1

-- If using GPUs, set the GPU before initializing MPI
local mpi = require('torchmpi')
mpi.start(config.gpu)
-- Experiment with these options to tune collective performance
if false then
   -- mpi.C.torchmpi_set_flat_collectives()
   -- OR
   mpi.C.torchmpi_set_hierarchical_collectives()

   -- mpi.C.torchmpi_set_staged_collectives()
   -- OR
   mpi.C.torchmpi_set_direct_collectives()

   mpi.C.torchmpi_set_cartesian_communicator()
   --OR
   -- mpi.C.torchmpi_set_tree_communicator()

   mpi.C.torchmpi_set_num_buffers_per_cpu_collective(4)
   mpi.C.torchmpi_set_num_buffers_per_gpu_collective(4)
   mpi.C.torchmpi_set_min_buffer_size_per_cpu_collective(2^10)
   mpi.C.torchmpi_set_max_buffer_size_per_cpu_collective(2^20)
   mpi.C.torchmpi_set_min_buffer_size_per_gpu_collective(2^10)
   mpi.C.torchmpi_set_max_buffer_size_per_gpu_collective(2^20)
   mpi.C.torchmpi_set_broadcast_size_cpu_tree_based(2^22)
   mpi.C.torchmpi_set_broadcast_size_gpu_tree_based(2^22)
end

local mpicache = require('torchmpi.cache')
local tester = require('torchmpi.tester')

local tests = {}

local asyncTimer = torch.Timer()

-------------------------------- broadcast --------------------------------
tests.broadcast = {}
-- only 1 tensor, no input/output distinction
tests.broadcast.test = function(input, output, firstRun)
   -- mpi.p2p, mpi.async.p2p
   local ns = config.async and mpi.async.p2p or mpi.p2p

   if config.async then
      asyncTimer = torch.Timer()
   end

   local ok, res = pcall(ns.broadcastTensor, mpi.size() - 1, input)
   if not ok then
      assert(res:find('NYI:'), res)
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

-- Careful, precision counts here once we reach a certain size
tests.broadcast.check = function(input) -- only 1 tensor, no input/output distinction
   -- 0-based
   local val = mpi.size() - 1
   local min, max = input:min(), input:max()
   if min ~= val or max ~= val then
      error(('[%d/%d] %f vs expected %f %s (size: %d)\n'):format(
            mpi.rank(), mpi.size(), min, val, input:data(), input:nElement()))
   end
end

-- Assumes a pipelined implementation of broadcast
tests.broadcast.communicationVolumeGB = function(t)
   local elemSize = 4
   return (t:nElement() * elemSize) / 1e9
end

-------------------------------- allreduce --------------------------------
tests.allreduce = {}
tests.allreduce.test = function(input, output)
   -- Output must be zeroed explicitly to get proper results, only when out of place
   if input ~= output then output:zero() end

   -- mpi.p2p, mpi.async.p2p
   local ns = config.async and mpi.async.p2p or mpi.p2p

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
   if min ~= val or max ~= val then
      error(('[%d/%d] %f-%f vs expected %f (size %d)\n'):format(
            mpi:rank(), mpi:size(), min, max, val, output:nElement()))
   end
end

-- Assumes a ring-based implementation
tests.allreduce.communicationVolumeGB = function(input)
   local elemSize = 4
   return (2 * input:nElement() * elemSize * (mpi.size() - 1) / mpi.size()) / 1e9
end

-------------------------------- Start tests --------------------------------
tests.broadcast.implemented = true
tests.allreduce.implemented = true

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
