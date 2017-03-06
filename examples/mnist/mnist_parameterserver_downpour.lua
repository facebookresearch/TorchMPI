--[[
 Copyright (c) 2016-present, Facebook, Inc.
 All rights reserved.

 This source code is licensed under the BSD-style license found in the
 LICENSE file in the root directory of this source tree. An additional grant
 of patent rights can be found in the PATENTS file in the same directory.
--]]
require('nn')
require('paths')

local tnt = require('torchnet')

-- use GPU or not:
local cmd = torch.CmdLine()
cmd:option('-usegpu', false, 'use gpu for training')
cmd:option('-seed', 1111, 'use gpu for training')
cmd:option('-prefetch', 5, 'prefetch distance for asynchronous communications')
cmd:option('-tau', 10, 'communication cycle length for parameterserver (see easgd paper, we reuse the notation)')
cmd:option('-tauInitDelay', 20, 'delay the first communication to let the networks search a bit independently')
cmd:option('-tauSendFrequency', 1, 'frequency at which we perform async sends')

local config = cmd:parse(arg)
print(string.format('running on %s', config.usegpu and 'GPU' or 'CPU'))

local mpi = require('torchmpi')
-- The model we use for GPU + MPI is 1 Lua/Terra process for 1 GPU
-- mpi.start sets the GPU automatically
mpi.start(config.usegpu)
local mpinn = require('torchmpi.nn')
local cache = require('torchmpi.cache')
local parameterserver = require('torchmpi.parameterserver')

-- Set the random seed manually for reproducibility.
torch.manualSeed(config.seed)

-- set up logistic regressor:
local net = nn.Sequential():add(nn.Linear(784,10))
local criterion = nn.CrossEntropyCriterion()

-- set up training engine:
local engine = tnt.SGDEngine()
local meter  = tnt.AverageValueMeter()
local clerr  = tnt.ClassErrorMeter{topk = {1}}
engine.hooks.onStartEpoch = function(state)
   meter:reset()
   clerr:reset()
end

engine.hooks.onForwardCriterion = function(state)
   meter:add(state.criterion.output)
   clerr:add(state.network.output, state.sample.target)
   if state.training then
      print(string.format('[%d/%d] avg. loss: %2.4f; avg. error: %2.4f',
         mpi.rank() + 1, mpi.size(), meter:value(), clerr:value{k = 1}))
   end
end

local initParameterServer = config.tauInitDelay
local nextPrefetch = config.tauInitDelay + config.tau + config.prefetch
local nextIntegration = config.tauInitDelay + config.tau
local nextSend = config.tauInitDelay + config.tauSendFrequency
assert(config.prefetch >= 0 and config.prefetch <= config.tau) -- prefetch needs to make sense ..

local handlesSend = {}
local handlesPrefetch = {}
engine.hooks.onBackward = function(state)
   assert(nextPrefetch >= state.t)    -- skipped a beat ..
   assert(nextIntegration >= state.t) -- skipped a beat ..
   if state.t == initParameterServer then
      local p, g  = state.network:parameters()
      parameterserver.initTensors(p)
   end
   if state.t == nextPrefetch then
      local p, g  = state.network:parameters()
      handlesPrefetch = parameterserver.prefetchTensors(p)
      nextPrefetch = nextPrefetch + config.tau
   end
   if state.t == nextIntegration then
      -- Make sure prefetches completed, you can also play with disabling this
      handlesPrefetch = parameterserver.syncHandles(handlesPrefetch)
      local p, g  = state.network:parameters()
      parameterserver.integrateTensors(p,
                                       function(pref, t) t:copy(pref) end)
      nextIntegration = nextIntegration + config.tau
   end
   if state.t == nextSend then
      local p, g  = state.network:parameters()
      handlesSend = parameterserver.sendTensors(
         p,
         g,
         'add',
         function(t) t:mul(-state.lr); return t end)
      handlesSend = parameterserver.syncHandles(handlesSend)
      nextSend = nextSend + config.tauSendFrequency
   end
   -- Disable torchnet.engine SGD step, downpour already took care of it
   -- TODO: do this better
   state.lrSave = state.lr
   state.lr = 0
end

-- 6. Reenable torchnet.engine SGD step with the last lr
-- TODO: do this better
engine.hooks.onUpdate = function(state)
   state.lr = state.lrSave
end

-- set up GPU training:
if config.usegpu then
   cutorch.manualSeed(config.seed)

   -- copy model to GPU:
   require('cunn')
   net       = net:cuda()
   criterion = criterion:cuda()

   -- copy sample to GPU buffer:
   local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
   engine.hooks.onSample = function(state)
      igpu:resize(state.sample.input:size() ):copy(state.sample.input)
      tgpu:resize(state.sample.target:size()):copy(state.sample.target)
      state.sample.input  = igpu
      state.sample.target = tgpu
   end  -- alternatively, this logic can be implemented via a TransformDataset
end

-- Perform weight and bias synchronization before starting training
mpinn.synchronizeParameters(net)
for _, v in pairs(net:parameters()) do mpi.checkWithAllreduce(v, 'initialParameters') end

local makeIterator = paths.dofile('makeiterator.lua')

-- train the model:
engine:train{
   network   = net,
   iterator  = makeIterator('train'),
   criterion = criterion,
   lr        = 0.2,
   maxepoch  = 5,
}

-- measure test loss and error:
meter:reset()
clerr:reset()
engine:test{
   network   = net,
   iterator  = makeIterator('test'),
   criterion = criterion,
}

-- Wait for all to finish before printing
mpi.barrier()

-- There is no real synchronization, checking allreduce does not makes sense
-- for _, v in pairs(net:parameters()) do mpi.checkWithAllreduce(v, 'final parameters') end

local loss = meter:value()
local err = clerr:value{k = 1}
print(string.format('[%d/%d] test loss: %2.4f; test error: %2.4f',
   mpi.rank() + 1, mpi.size(), loss, err))

-- There is no real synchronization, checking allreduce does not makes sense
-- mpi.checkWithAllreduce(loss, 'final loss')

mpi.stop()
