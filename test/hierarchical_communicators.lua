--[[
 Copyright (c) 2016-present, Facebook, Inc.
 All rights reserved.

 This source code is licensed under the BSD-style license found in the
 LICENSE file in the root directory of this source tree. An additional grant
 of patent rights can be found in the PATENTS file in the same directory.
--]]
require('torch')

local mpi = require('torchmpi')
local mpicache = require('torchmpi.cache')

-- This is an example of setting up a custom communicator say for performing
-- parameter server operations on top of synchronous SGD communicators.
-- A customCommunicatorInit function should return the result of
-- mpi.C.torchmpi_push_communicator with your preferred string for discriminating
-- amongst participants.
-- Processes with the same string will end up in the same (intra) group.
-- Processes with rank 0 within each group form another (inter) group.
--
-- Run like this:
-- mpirun -n 32 --map-by node --bind-to none --hostfile /etc/JARVICE/nodes luajit ./test/hierarchical_communicators.lua

local function customCommunicatorInit()
   local res =
      mpi.C.torchmpi_push_communicator(tostring(math.floor(mpi.rank() % 3)));
   assert(res == 1)
   return res
end

mpi.start(false, customCommunicatorInit)

local rankG = mpi.rank()

-- Creating a custom communicator will leave you at the level it has been created
-- Get back to level 0 to do stuff form the top.
mpi.C.torchmpi_set_communicator(0)

mpi.barrier()
if mpi.rank() == 0 then print('Success') end
mpi.stop()
