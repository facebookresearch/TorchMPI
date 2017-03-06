/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#include "collectives_cuda.h"
#include "collectives.h"

#include <algorithm>
#include <iostream>
#include <unordered_map>

#include "resources.h"

/**********************************************************************
 ******************************** MPI *********************************
 **********************************************************************/
using namespace std;
using namespace torch::mpi;
using namespace torch::mpi::constants;
using namespace torch::mpi::resources;
using namespace torch::mpi::resources::cuda;

namespace torch { namespace mpi { namespace thc {

#define PREPARE(state, tensor)                                          \
  THCudaCheck(cudaGetLastError());                                      \
  if (!torch::thc::isContiguous(state, tensor)) {                       \
    THError("NYI: Sendrecv_replace only supported for contig tensors"); \
  }                                                                     \
  int device;                                                           \
  THCudaCheck(cudaGetDevice(&device));                                  \
  torch::mpi::thc::retainStorage(state, tensor);                        \
  auto stream = THCState_getCurrentStream(state);                       \
  auto tensorData = torch::thc::data<ScalarType>(state, tensor);        \
  auto nElement = torch::thc::nElement<THTensorType>(state, tensor);    \
  auto collectiveLevel = torch::mpi::getCollectiveSpan().first;         \
  CommunicatorGuard csOuter(collectiveLevel);                           \
  const CollectiveResources* rOuter = acquireCollectiveResources(       \
    tensorData, Spin(true));

#define PREPARE_NCCL(state, tensor)                                     \
  PREPARE(state, tensor);                                               \
  CommunicatorGuard csInner(getCollectiveSpan().second);                \
  const CollectiveResources* rInner =                                   \
    acquireCollectiveResources(tensorData,                              \
                               Spin(true),                              \
                               WithNCCLComm(true));                     \
  auto hasIntra = getMainThreadCommunicator().hasIntraCollective();     \
  auto hasInter = getMainThreadCommunicator().hasInterCollective();

#define PREPARE_IPC(state, tensor)                                      \
  PREPARE(state, tensor);                                               \
  CommunicatorGuard csInner(getCollectiveSpan().second);                \
  const CollectiveResources* rInner = acquireCollectiveResources(       \
    tensorData, Spin(true), WithNCCLComm(false), WithEvents(true));     \
  auto tensorDataBasePtr =                                              \
    torch::thc::data<ScalarType, THTensorType>(state, tensor) -         \
    tensor->storageOffset;                                              \
  auto desc =                                                           \
    getIPCDesc(state, tensorDataBasePtr, rInner->comm->intraComm);      \
  auto hasIntra = getMainThreadCommunicator().hasIntraCollective();     \
  auto hasInter = getMainThreadCommunicator().hasInterCollective();

#define PREPARE2(state, input, output)                                  \
  THCudaCheck(cudaGetLastError());                                      \
  if (!torch::thc::isContiguous(state, input)) {                        \
    THError("NYI: Reduce only supported for contig tensors");           \
  }                                                                     \
  torch::mpi::thc::retainStorage(state, input);                         \
  if (input != output) {                                                \
    torch::mpi::thc::retainStorage(state, output);                      \
  }                                                                     \
  int device;                                                           \
  THCudaCheck(cudaGetDevice(&device));                                  \
  auto stream = THCState_getCurrentStream(state);                       \
  auto inputData = torch::thc::data<ScalarType>(state, input);          \
  auto outputData = (output) ?                                          \
    torch::thc::data<ScalarType>(state, output) : inputData;            \
  auto nElement = torch::thc::nElement<THTensorType>(state, input);     \
  auto collectiveLevel = getCollectiveSpan().first;                     \
  CommunicatorGuard cs(collectiveLevel);                                \
  const CollectiveResources* rOuter = acquireCollectiveResources(       \
    inputData, Spin(true));

#define PREPARE2_NCCL(state, input, output)                             \
  PREPARE2(state, input, output);                                       \
  CommunicatorGuard csInner(getCollectiveSpan().second);                \
  const CollectiveResources* rInner =                                   \
    acquireCollectiveResources(inputData,                               \
                               Spin(true),                              \
                               WithNCCLComm(true));                     \
  auto hasIntra = getMainThreadCommunicator().hasIntraCollective();     \
  auto hasInter = getMainThreadCommunicator().hasInterCollective();

#define PREPARE2_IPC(state, input, output)                              \
  PREPARE2(state, input, output);                                       \
  CommunicatorGuard csInner(getCollectiveSpan().second);                \
  const CollectiveResources* rInner = acquireCollectiveResources(       \
    inputData, Spin(true), WithNCCLComm(false), WithEvents(true));      \
  auto outputDataBasePtr =                                              \
    torch::thc::data<ScalarType, THTensorType>(state, output) -         \
    output->storageOffset;                                              \
  auto desc =                                                           \
    getIPCDesc(state, outputDataBasePtr, rInner->comm->intraComm);      \
  auto hasIntra = getMainThreadCommunicator().hasIntraCollective();     \
  auto hasInter = getMainThreadCommunicator().hasInterCollective();


std::vector<cudaStream_t> preSyncHiPriStreams(cudaStream_t s) {
  auto p = getCollectiveStreams();
  THCudaCheck(cudaEventRecord(p.first, s));
  for (auto stream : p.second) {
    THCudaCheck(cudaStreamWaitEvent(stream, p.first, 0));
  }
  return p.second;
}

void postSyncHiPriStreams(cudaStream_t s) {
  auto p = getCollectiveStreams();
  for (auto stream : p.second) {
    THCudaCheck(cudaEventRecord(p.first, stream));
  }
  THCudaCheck(cudaStreamWaitEvent(s, p.first, 0));
}


///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//                  Blocking collectives.                                    //
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
template<typename ScalarType>
void broadcast(ScalarType* inputData,
               int root,
               size_t nElement,
               const CollectiveResources* r)
{
  r->comm->intraComm.Bcast(inputData, nElement, mpiType<ScalarType>(), root);
}

template<typename ScalarType>
void allreduce(ScalarType* inputData,
               ScalarType* outputData,
               size_t nElement,
               MPI::Op mpiRedOp,
               const CollectiveResources* r)
{
  r->comm->intraComm.Allreduce(
    (outputData != inputData) ? inputData : MPI_IN_PLACE,
    outputData,
    nElement,
    mpiType<ScalarType>(),
    mpiRedOp);
}


template<typename ScalarType, typename THTensorType>
void sendreceive(THCState* state, THTensorType* tensor, int src, int dst) {
  PREPARE(state, tensor);

  rOuter->comm->intraComm.Sendrecv_replace(tensorData,
                                 nElement,
                                 mpiType<ScalarType>(),
                                 dst,
                                 kDefaultTag,
                                 src,
                                 kDefaultTag);

  // TODO: ScopeGuard
  releaseCollectiveResources(const_cast<CollectiveResources*>(rOuter));

  THCudaCheck(cudaGetLastError());
}

template<typename ScalarType, typename THTensorType>
void broadcast(THCState* state,
               THTensorType* tensor,
               int root)
{
  PREPARE(state, tensor);

  broadcast<ScalarType>(tensorData, root, nElement, rOuter);

  // TODO: ScopeGuard
  releaseCollectiveResources(const_cast<CollectiveResources*>(rOuter));

  THCudaCheck(cudaGetLastError());
}

template<typename ScalarType, typename THTensorType>
void reduce(THCState* state,
            THTensorType* input,
            THTensorType* output,
            int root,
            MPI::Op mpiRedOp)
{
  PREPARE2(state, input, output);

  if (outputData == inputData) {
    rOuter->comm->intraComm.Reduce(
      (commRank(rOuter->comm->intraComm) == root) ? MPI_IN_PLACE : inputData,
      outputData,
      nElement,
      mpiType<ScalarType>(),
      mpiRedOp,
      root);
  } else {
    rOuter->comm->intraComm.Reduce(
      inputData,
      outputData,
      nElement,
      mpiType<ScalarType>(),
      mpiRedOp,
      root);
  }

  // TODO: ScopeGuard
  releaseCollectiveResources(const_cast<CollectiveResources*>(rOuter));

  THCudaCheck(cudaGetLastError());
}

template<typename ScalarType, typename THTensorType>
void allreduce(THCState* state,
               THTensorType* input,
               THTensorType* output,
               MPI::Op mpiRedOp)
{
  PREPARE2(state, input, output);

  allreduce(inputData, outputData, nElement, mpiRedOp, rOuter);

  // TODO: ScopeGuard
  releaseCollectiveResources(const_cast<CollectiveResources*>(rOuter));

  THCudaCheck(cudaGetLastError());
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//      P2P collectives perform barriers internally                          //
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
template<typename ScalarType>
SynchronizationHandle* broadcastp2pIPCImpl(ScalarType* dataPtr,
                                           int root,
                                           size_t nElement,
                                           const MPI::Intracomm& comm,
                                           IPCDesc* desc,
                                           size_t offset,
                                           cudaStream_t stream,
                                           const CollectiveIpcEvents &events)
{
  auto hiPriStreams = preSyncHiPriStreams(stream);
  THAssert(hiPriStreams.size() > 0);
  auto hiPriStream = hiPriStreams[0];
  if (nElement >= constants::kSmallBcastSizeGPU) {
    detail::broadcastp2pIPC<ScalarType>(
      dataPtr,
      root,
      nElement,
      desc,
      offset,
      comm,
      hiPriStream,
      events
    );
  } else {
    // TODO: Would be better to just use stock MPI here but for some reason,
    // I see registration error messages in this particular case
    auto b = SmallPinnedBufferProvider::acquire();
    if (root == commRank(comm)) {
      b->copyFrom(dataPtr, nElement * sizeof(ScalarType), hiPriStream);
      // Must sync to avoid Bcast too early
      THCudaCheck(cudaStreamSynchronize(hiPriStream));
    }
    comm.Bcast(b->data, nElement, mpiType<ScalarType>(), root);
    if (root != commRank(comm)) {
      b->copyTo(dataPtr, nElement * sizeof(ScalarType), hiPriStream);
      // Must sync to avoid releasing buffer too early
      THCudaCheck(cudaStreamSynchronize(hiPriStream));
    }
    SmallPinnedBufferProvider::release(b);
   }
  postSyncHiPriStreams(stream);
  return synchronizationHandleFromStream(stream);
}

template<typename ScalarType, typename THTensorType>
void broadcastp2p(THCState* state,
                  THTensorType* tensor,
                  int root)
{
  PREPARE_IPC(state, tensor);
  if (hasInter) {
    // TODO: ScopeGuard
    releaseCollectiveResources(const_cast<CollectiveResources*>(rInner));
    releaseCollectiveResources(const_cast<CollectiveResources*>(rOuter));
    THError("NYI: Multi-node/IPC domain P2P broadcast not yet supported, " \
            "use the stock MPI broadcast");
  }
  auto sh = broadcastp2pIPCImpl<ScalarType>(tensorData,
                                            root,
                                            nElement,
                                            rInner->comm->intraComm,
                                            desc,
                                            tensor->storageOffset,
                                            stream,
                                            rInner->events);
  THCudaCheck(cudaGetLastError());
  resources::wait(sh);

  // TODO: ScopeGuard
  releaseCollectiveResources(const_cast<CollectiveResources*>(rInner));
  releaseCollectiveResources(const_cast<CollectiveResources*>(rOuter));
}

template<typename ScalarType>
SynchronizationHandle* allreducep2pIPCImpl(ScalarType* inputData,
                                           ScalarType* outputData,
                                           size_t nElement,
                                           MPI::Op mpiRedOp,
                                           const MPI::Intracomm& comm,
                                           IPCDesc* desc,
                                           size_t offset,
                                           cudaStream_t stream,
                                           const CollectiveIpcEvents &events)
{
  auto hiPriStreams = preSyncHiPriStreams(stream);
  THAssert(hiPriStreams.size() > 0);
  auto hiPriStream = hiPriStreams[0];
  if (nElement >= constants::kSmallAllreduceSizeGPU) {
    // This performs collective calls internally
    detail::allreducep2pIPC<ScalarType>(
      inputData,
      outputData,
      nElement,
      mpiRedOp,
      desc,
      offset,
      comm,
      hiPriStreams,
      events
    );
  } else {
    // TODO: Would be better to just use stock MPI here but for some reason,
    // I see registration error messages in this particular case
    auto b = SmallPinnedBufferProvider::acquire();
    b->copyFrom(inputData, nElement * sizeof(ScalarType), hiPriStream);
    // Must sync to avoid Allreduce too early
    THCudaCheck(cudaStreamSynchronize(hiPriStream));
    comm.Allreduce(
      MPI_IN_PLACE, b->data, nElement, mpiType<ScalarType>(), mpiRedOp);
    b->copyTo(outputData, nElement * sizeof(ScalarType), hiPriStream);
    // Must sync to avoid releasing buffer too early
    THCudaCheck(cudaStreamSynchronize(hiPriStream));
    SmallPinnedBufferProvider::release(b);
  }
  postSyncHiPriStreams(stream);
  return synchronizationHandleFromStream(stream);
}

template<typename ScalarType>
SynchronizationHandle* allreducep2pHierarchicalImpl(
    ScalarType* inputData,
    ScalarType* outputData,
    size_t nElement,
    MPI::Op mpiRedOp,
    const MPI::Intracomm& ipcComm,
    const MPI::Intracomm& interIPCComm,
    IPCDesc* ipcDesc,
    size_t offset,
    cudaStream_t stream,
    bool hasIntra,
    bool hasInter,
    const CollectiveIpcEvents &events)
{
  // Short P2P path
  if (!hasInter && hasIntra) {
    allreducep2pIPCImpl<ScalarType>(inputData,
                                    outputData,
                                    nElement,
                                    mpiRedOp,
                                    ipcComm,
                                    ipcDesc,
                                    offset,
                                    stream,
                                    events);
    // don't sync on stream here
    return synchronizationHandleFromStream(stream);
  }

  // If we get here we must go hierarchical
  if (hasIntra) {
    allreducep2pIPCImpl<ScalarType>(inputData,
                                    outputData,
                                    nElement,
                                    mpiRedOp,
                                    ipcComm,
                                    ipcDesc,
                                    offset,
                                    stream,
                                    events);
    // don't sync on stream here
  }

  if (torch::mpi::commSize(interIPCComm) > 1) {
    auto hiPriStreams = torch::mpi::thc::preSyncHiPriStreams(stream);
    THAssert(hiPriStreams.size() > 0);
    torch::mpi::thc::detail::allreducep2pCrossNodes<ScalarType>(
      hasIntra ? outputData : inputData,
      outputData,
      nElement,
      mpiRedOp,
      interIPCComm,
      hiPriStreams);
    torch::mpi::thc::postSyncHiPriStreams(stream);
    // don't sync on stream here
  }
  if (hasIntra) {
    broadcastp2pIPCImpl(outputData,
                        0,
                        nElement,
                        ipcComm,
                        ipcDesc,
                        offset,
                        stream,
                        events);
    // don't sync on stream here
  }
  return synchronizationHandleFromStream(stream);
}

template<typename ScalarType, typename THTensorType>
void allreducep2pHierarchical(THCState* state,
                              THTensorType* input,
                              THTensorType* output,
                              MPI::Op mpiRedOp)
{
  PREPARE2_IPC(state, input, output);

  auto sh = allreducep2pHierarchicalImpl(inputData,
                                         outputData,
                                         nElement,
                                         mpiRedOp,
                                         rInner->comm->intraComm,
                                         rInner->comm->interComm,
                                         desc,
                                         output->storageOffset,
                                         stream,
                                         hasIntra,
                                         hasInter,
                                         rInner->events);
  // TODO: ScopeGuard??
  releaseCollectiveResources(const_cast<CollectiveResources*>(rInner));
  releaseCollectiveResources(const_cast<CollectiveResources*>(rOuter));

  // Must sync on stream here
  resources::wait(sh);
}

template<typename ScalarType, typename THTensorType>
void allreducep2pFlat(THCState* state,
                      THTensorType* input,
                      THTensorType* output,
                      MPI::Op mpiRedOp)
{
  PREPARE2(state, input, output);

  auto hiPriStreams = torch::mpi::thc::preSyncHiPriStreams(stream);
  THAssert(hiPriStreams.size() > 0);
  torch::mpi::thc::detail::allreducep2pCrossNodes<ScalarType>(
    inputData,
    outputData,
    nElement,
    mpiRedOp,
    rOuter->comm->intraComm,
    hiPriStreams);
  torch::mpi::thc::postSyncHiPriStreams(stream);

  // TODO: ScopeGuard??
  releaseCollectiveResources(const_cast<CollectiveResources*>(rOuter));

  // Must sync on stream here
  THCudaCheck(cudaStreamSynchronize(stream));
}

template<typename ScalarType, typename THTensorType>
void allreducep2p(THCState* state,
                  THTensorType* input,
                  THTensorType* output,
                  MPI::Op mpiRedOp)
{
  if (constants::kUseHierarchicalCollectives) {
    // If we have to go through TCP we cannot rely on cuda support to properly
    // perform CPU-GPU copies asynchronously. Write our own hierarchical
    // allreduce that goes through explicit copies ot pinned CPU buffers.
    allreducep2pHierarchical<ScalarType, THTensorType>(
      state, input, output, mpiRedOp);
  } else{
    // 1-level flat and simple Allreduce using MPI_Isend/MPI_Irecv backed by
    // RDMA.
    allreducep2pFlat<ScalarType, THTensorType>(
      state, input, output, mpiRedOp);
  }
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//   All async collectives offload to the collectiveOffloadThreadPool        //
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template<typename ScalarType, typename THTensorType>
SynchronizationHandle* broadcastAsync(
    THCState* state, THTensorType* tensor, int root) {
  PREPARE(state, tensor);
  auto& futures = getCollectiveFutures();
  futures.push_back(
    collectiveOffloadThreadPool().enqueue([=](){
      THCudaCheck(cudaSetDevice(device));
      broadcast<ScalarType>(tensorData, root, nElement, rOuter);
      THCudaCheck(cudaGetLastError());
      // TODO: ScopeGuard??
      releaseCollectiveResources(const_cast<CollectiveResources*>(rOuter));
  }));
  return synchronizationHandleFromFuture(futures.size() - 1);
}

template<typename ScalarType, typename THTensorType>
SynchronizationHandle* allreduceAsync(
  THCState* state, THTensorType* input, THTensorType* output, MPI::Op mpiRedOp)
{
  PREPARE2(state, input, output);
  auto& futures = getCollectiveFutures();
  futures.push_back(
    collectiveOffloadThreadPool().enqueue([=](){
      THCudaCheck(cudaSetDevice(device));
      allreduce<ScalarType>(
        inputData, outputData, nElement, mpiRedOp, rOuter);
      THCudaCheck(cudaGetLastError());
      // TODO: ScopeGuard??
      releaseCollectiveResources(const_cast<CollectiveResources*>(rOuter));
    }));
  return synchronizationHandleFromFuture(futures.size() - 1);
}

template<typename ScalarType>
SynchronizationHandle* broadcastp2pIPCAsyncImpl(
    ScalarType* tensorData,
    int root,
    size_t nElement,
    const CollectiveResources* r,
    IPCDesc* desc,
    size_t offset,
    cudaStream_t stream) {
  int device;
  THCudaCheck(cudaGetDevice(&device));
  auto& futures = getCollectiveFutures();
  futures.push_back(
    collectiveOffloadThreadPool().enqueue([=](){
      THCudaCheck(cudaSetDevice(device));
      auto sh = broadcastp2pIPCImpl<ScalarType>(tensorData,
                                                root,
                                                nElement,
                                                r->comm->intraComm,
                                                desc,
                                                offset,
                                                stream,
                                                r->events);
      // Must sync on stream here
      resources::wait(sh);
      // TODO: ScopeGuard
      releaseCollectiveResources(const_cast<CollectiveResources*>(r));
  }));
  return synchronizationHandleFromFuture(futures.size() - 1);
}

template<typename ScalarType, typename THTensorType>
SynchronizationHandle* broadcastp2pAsync(THCState* state,
                                         THTensorType* tensor,
                                         int root)
{
  PREPARE_IPC(state, tensor);
  if (hasInter) {
    // TODO: ScopeGuard
    releaseCollectiveResources(const_cast<CollectiveResources*>(rInner));
    releaseCollectiveResources(const_cast<CollectiveResources*>(rOuter));
    THError("NYI: Multi-node/IPC domain broadcast not yet supported, "  \
            "use the stock MPI broadcast");
  }
  // broadcastp2pIPCAsyncImpl must release rInner!!
  releaseCollectiveResources(const_cast<CollectiveResources*>(rOuter));
  return broadcastp2pIPCAsyncImpl<ScalarType>(tensorData,
                                              root,
                                              nElement,
                                              rInner,
                                              desc,
                                              tensor->storageOffset,
                                              stream);
}

template<typename ScalarType, typename THTensorType>
SynchronizationHandle* allreducep2pAsyncHierarchical(
  THCState* state, THTensorType* input, THTensorType* output, MPI::Op mpiRedOp)
{
  PREPARE2_IPC(state, input, output);

  auto& futures = getCollectiveFutures();
  futures.push_back(
    collectiveOffloadThreadPool().enqueue([=](){
      THCudaCheck(cudaSetDevice(device));

      // 2-level implementation going through cudaIPC for intra-node
      // and explicit pinned CPU buffers for inter-node.
      auto sh = allreducep2pHierarchicalImpl(inputData,
                                             outputData,
                                             nElement,
                                             mpiRedOp,
                                             rInner->comm->intraComm,
                                             rInner->comm->interComm,
                                             desc,
                                             output->storageOffset,
                                             stream,
                                             hasIntra,
                                             hasInter,
                                             rInner->events);
      // Must sync on stream here
      resources::wait(sh);

      // TODO: ScopeGuard??
      releaseCollectiveResources(const_cast<CollectiveResources*>(rInner));
      releaseCollectiveResources(const_cast<CollectiveResources*>(rOuter));
  }));

  return synchronizationHandleFromFuture(futures.size() - 1);
}


template<typename ScalarType, typename THTensorType>
SynchronizationHandle* allreducep2pAsyncFlat(
  THCState* state, THTensorType* input, THTensorType* output, MPI::Op mpiRedOp)
{
  PREPARE2_IPC(state, input, output);

  auto& futures = getCollectiveFutures();
  futures.push_back(
    collectiveOffloadThreadPool().enqueue([=](){
      THCudaCheck(cudaSetDevice(device));

      auto hiPriStreams = torch::mpi::thc::preSyncHiPriStreams(stream);
      THAssert(hiPriStreams.size() > 0);
      torch::mpi::thc::detail::allreducep2pCrossNodes<ScalarType>(
        inputData,
        outputData,
        nElement,
        mpiRedOp,
        rOuter->comm->intraComm,
        hiPriStreams);
      torch::mpi::thc::postSyncHiPriStreams(stream);

      // Must sync on stream here
      THCudaCheck(cudaStreamSynchronize(stream));

      // TODO: ScopeGuard??
      releaseCollectiveResources(const_cast<CollectiveResources*>(rInner));
      releaseCollectiveResources(const_cast<CollectiveResources*>(rOuter));
  }));

  return synchronizationHandleFromFuture(futures.size() - 1);
}


template<typename ScalarType, typename THTensorType>
SynchronizationHandle* allreducep2pAsync(
  THCState* state, THTensorType* input, THTensorType* output, MPI::Op mpiRedOp)
{
  if (constants::kUseHierarchicalCollectives) {
    // If we have to go through TCP we cannot rely on cuda support to properly
    // perform CPU-GPU copies asynchronously. Write our own hierarchical
    // allreduce that goes through explicit copies ot pinned CPU buffers.
    return allreducep2pAsyncHierarchical<ScalarType, THTensorType>(
      state, input, output, mpiRedOp);
  } else {
    // 1-level flat and simple Allreduce using MPI_Isend/MPI_Irecv backed by
    // RDMA.
    return allreducep2pAsyncFlat<ScalarType, THTensorType>(
      state, input, output, mpiRedOp);
  }
}

}} // ns mpi::thc


#ifdef TORCH_MPI_NCCL

namespace nccl { namespace thc {

// Collectives operating on THCuda*Tensor
template<typename ScalarType>
cudaStream_t broadcast(ScalarType* tensorData,
                       int root,
                       size_t nElement,
                       cudaStream_t stream,
                       const ncclComm_t& comm)
{
  int count, device, rank;
  NCCLCHECK(ncclCommCount(comm, &count));
  NCCLCHECK(ncclCommCuDevice(comm, &device));
  NCCLCHECK(ncclCommUserRank(comm, &rank));
  VLOG_1(" ncclBroadcast device " << device << " rank " << rank
         << "/" << count << std::endl);
  NCCLCHECK(ncclBcast(tensorData,
                      nElement,
                      ncclType<ScalarType>(),
                      root,
                      comm,
                      stream));
  THCudaCheck(cudaGetLastError());
  return stream;
}

// Collectives operating on THCuda*Tensor
template<typename ScalarType, typename THTensorType>
cudaStream_t broadcastImpl(THCState* state, THTensorType* tensor, int root) {
  PREPARE_NCCL(state, tensor);
  if (hasInter) {
    // TODO: ScopeGuard
    releaseCollectiveResources(const_cast<CollectiveResources*>(rInner));
    releaseCollectiveResources(const_cast<CollectiveResources*>(rOuter));
    THError("NYI: Multi-node/IPC domain NCCL broadcast not yet supported, " \
            "use the stock MPI broadcast");
  }
  nccl::thc::broadcast(tensorData, root, nElement, stream, *rInner->ncclComm);
  // TODO: ScopeGuard
  releaseCollectiveResources(const_cast<CollectiveResources*>(rInner));
  releaseCollectiveResources(const_cast<CollectiveResources*>(rOuter));
  return stream;
}

template<typename ScalarType, typename THTensorType>
void broadcast(THCState* state, THTensorType* tensor, int root) {
  THCudaCheck(cudaStreamSynchronize(
    nccl::thc::broadcastImpl<ScalarType, THTensorType>(
      state, tensor, root)));
}

template<typename ScalarType, typename THTensorType>
SynchronizationHandle*
broadcastAsync(THCState* state, THTensorType* tensor, int root) {
  return synchronizationHandleFromStream(
    nccl::thc::broadcastImpl<ScalarType, THTensorType>(
      state, tensor, root));
}


// Collectives operating on THCuda*Tensor
template<typename ScalarType>
cudaStream_t reduce(ScalarType* inputData,
                    ScalarType* outputData,
                    int root,
                    size_t nElement,
                    ncclRedOp_t ncclRedOp,
                    cudaStream_t stream,
                    const ncclComm_t& comm)
{
  int count, device, rank;
  NCCLCHECK(ncclCommCount(comm, &count));
  NCCLCHECK(ncclCommCuDevice(comm, &device));
  NCCLCHECK(ncclCommUserRank(comm, &rank));
  VLOG_1(" ncclReduce device " << device << " rank " << rank
         << "/" << count << std::endl);
  NCCLCHECK(ncclReduce(inputData,
                       outputData,
                       nElement,
                       ncclType<ScalarType>(),
                       ncclRedOp,
                       root,
                       comm,
                       stream));
  THCudaCheck(cudaGetLastError());
  return stream;
}

// Collectives operating on THCuda*Tensor
template<typename ScalarType, typename THTensorType>
cudaStream_t reduceImpl(THCState* state,
                        THTensorType* input,
                        THTensorType* output,
                        int root,
                        ncclRedOp_t ncclRedOp)
{
  PREPARE2_NCCL(state, input, output);
  if (hasInter) {
    // TODO: ScopeGuard
    releaseCollectiveResources(const_cast<CollectiveResources*>(rInner));
    releaseCollectiveResources(const_cast<CollectiveResources*>(rOuter));
    THError("NYI: Multi-node/IPC domain NCCL reduce not yet supported, " \
            "use the stock MPI broadcast");
  }
  // Just reduce within node level, the NCCL communicator is unique
  nccl::thc::reduce(
    inputData, outputData, root, nElement, ncclRedOp, stream, *rInner->ncclComm);
  // TODO: ScopeGuard
  releaseCollectiveResources(const_cast<CollectiveResources*>(rInner));
  releaseCollectiveResources(const_cast<CollectiveResources*>(rOuter));
  return stream;
}

template<typename ScalarType, typename THTensorType>
void reduce(THCState* state,
            THTensorType* input,
            THTensorType* output,
            int root,
            ncclRedOp_t ncclRedOp) {
  THCudaCheck(cudaStreamSynchronize(reduceImpl<ScalarType, THTensorType>(
    state, input, output, root, ncclRedOp)));
}

template<typename ScalarType, typename THTensorType>
SynchronizationHandle* reduceAsync(THCState* state,
                                   THTensorType* input,
                                   THTensorType* output,
                                   int root,
                                   ncclRedOp_t ncclRedOp) {
  return synchronizationHandleFromStream(reduceImpl<ScalarType, THTensorType>(
    state, input, output, root, ncclRedOp));
}


template<typename ScalarType>
cudaStream_t allreduce(ScalarType* inputData,
                       ScalarType* outputData,
                       size_t nElement,
                       ncclRedOp_t ncclRedOp,
                       cudaStream_t stream,
                       const ncclComm_t& comm)
{
  int count, device, rank;
  NCCLCHECK(ncclCommCount(comm, &count));
  NCCLCHECK(ncclCommCuDevice(comm, &device));
  NCCLCHECK(ncclCommUserRank(comm, &rank));
  VLOG_1(" ncclAllReduce device " << device << " rank " << rank
         << "/" << count << std::endl);
  NCCLCHECK(ncclAllReduce(inputData,
                          outputData,
                          nElement,
                          ncclType<ScalarType>(),
                          ncclRedOp,
                          comm,
                          stream));
  THCudaCheck(cudaGetLastError());
  return stream;
}

template<typename ScalarType, typename THTensorType>
SynchronizationHandle* allreduceImpl(THCState* state,
                                     THTensorType* input,
                                     THTensorType* output,
                                     ncclRedOp_t ncclRedOp)
{
  PREPARE2_NCCL(state, input, output);

  // Case 1. Intra only
  if (!hasInter) {
    // If we don't go cross NCCL invocations, never offload to helper thread
    // because we have no guarantee when kernels will actually get posted.
    // In this case asynchronicity should be handled by streams only
    auto res = synchronizationHandleFromStream(
      nccl::thc::allreduce<ScalarType>(inputData,
                                       outputData,
                                       nElement,
                                       ncclRedOp,
                                       stream,
                                       *rInner->ncclComm));
    // TODO: ScopeGuard
    releaseCollectiveResources(const_cast<CollectiveResources*>(rInner));
    releaseCollectiveResources(const_cast<CollectiveResources*>(rOuter));
    return res;
  }

  // Case 2. Intra only
  if (!hasIntra) {
    auto hiPriStreams = torch::mpi::thc::preSyncHiPriStreams(stream);
    THAssert(hiPriStreams.size() > 0);
    torch::mpi::thc::detail::allreducep2pCrossNodes<ScalarType>(
      hasIntra ? outputData : inputData,
      outputData,
      nElement,
      mpiOp(ncclRedOp),
      rInner->comm->interComm,
      hiPriStreams);
    torch::mpi::thc::postSyncHiPriStreams(stream);
    // TODO: ScopeGuard
    releaseCollectiveResources(const_cast<CollectiveResources*>(rInner));
    releaseCollectiveResources(const_cast<CollectiveResources*>(rOuter));
    return nullptr;
  }

  // Case 3. Both inter and intra
  auto lambda = [=]() {
      nccl::thc::allreduce<ScalarType>(inputData,
                                       outputData,
                                       nElement,
                                       ncclRedOp,
                                       stream,
                                       *rInner->ncclComm);
      auto hiPriStreams = torch::mpi::thc::preSyncHiPriStreams(stream);
      THAssert(hiPriStreams.size() > 0);
      torch::mpi::thc::detail::allreducep2pCrossNodes<ScalarType>(
        hasIntra ? outputData : inputData,
        outputData,
        nElement,
        mpiOp(ncclRedOp),
        rInner->comm->interComm,
        hiPriStreams);
      torch::mpi::thc::postSyncHiPriStreams(stream);
      nccl::thc::broadcast(outputData,
                           0,
                           nElement,
                           stream,
                           *rInner->ncclComm);
      THCudaCheck(cudaStreamSynchronize(stream));
      // TODO: ScopeGuard
      releaseCollectiveResources(const_cast<CollectiveResources*>(rInner));
      releaseCollectiveResources(const_cast<CollectiveResources*>(rOuter));
  };

  // Unfortunately trying to mix threads and multiple NCCL communicators
  // seems to create deadlocks.
  // So it seems no MPI overlapping of NCCL transfers with copies?
  // auto& futures = getCollectiveFutures();
  // futures.push_back(collectiveOffloadThreadPool().enqueue(l));
  // return synchronizationHandleFromFuture(futures.size() - 1);

  lambda();
  return synchronizationHandleFromStream(stream);
}

template<typename ScalarType, typename THTensorType>
void allreduce(THCState* state,
               THTensorType* input,
               THTensorType* output,
               ncclRedOp_t ncclRedOp) {
  resources::wait(
    nccl::thc::allreduceImpl<ScalarType, THTensorType>(
      state, input, output, ncclRedOp));

}

template<typename ScalarType, typename THTensorType>
SynchronizationHandle* allreduceAsync(THCState* state,
                                      THTensorType* input,
                                      THTensorType* output,
                                      ncclRedOp_t ncclRedOp) {
  return nccl::thc::allreduceImpl<ScalarType, THTensorType>(
    state, input, output, ncclRedOp);
}


}} // ns nccl::thc

#endif

} // ns torch



/**********************************************************************
 *********************** C Wrapper definitions ************************
 **********************************************************************/
#define PPCAT_NX(A, B) A ## B
#define PPCAT(A, B) PPCAT_NX(A, B)

extern "C" {

/*********************** Broadcast ************************************/
#define DEFINE_BROADCAST(ScalarType, THCTensorType)                     \
  void PPCAT(torchmpi_broadcast_, THCTensorType)                        \
    (THCState* state, THCTensorType *input, int root) {                 \
    struct cudaPointerAttributes attributes;                            \
    THCudaCheck(cudaPointerGetAttributes(&attributes,                   \
      torch::thc::data<ScalarType, THCTensorType>(state, input)));      \
    torch::mpi::thc::broadcast<ScalarType, THCTensorType>(              \
      state, input, root);                                              \
  }

#define DEFINE_BROADCAST_ASYNC(ScalarType, THCTensorType)               \
  SynchronizationHandle*                                                \
  PPCAT(torchmpi_async_broadcast_, THCTensorType)                       \
    (THCState* state, THCTensorType *input, int root) {                 \
  struct cudaPointerAttributes attributes;                              \
  THCudaCheck(cudaPointerGetAttributes(&attributes,                     \
    torch::thc::data<ScalarType, THCTensorType>(state, input)));        \
  return torch::mpi::thc::broadcastAsync<ScalarType, THCTensorType>(    \
  state, input, root);                                                  \
}

#define DEFINE_BROADCASTP2P(ScalarType, THCTensorType)                  \
  void PPCAT(torchmpi_p2p_broadcast_, THCTensorType)                    \
    (THCState* state, THCTensorType *input, int root) {                 \
    struct cudaPointerAttributes attributes;                            \
    THCudaCheck(cudaPointerGetAttributes(&attributes,                   \
      torch::thc::data<ScalarType, THCTensorType>(state, input)));      \
    torch::mpi::thc::broadcastp2p<ScalarType, THCTensorType>(           \
      state, input, root);                                              \
  }

#define DEFINE_BROADCASTP2P_ASYNC(ScalarType, THCTensorType)            \
  SynchronizationHandle* PPCAT                                          \
  (torchmpi_async_p2p_broadcast_, THCTensorType)                        \
    (THCState* state, THCTensorType *input, int root) {                 \
  struct cudaPointerAttributes attributes;                              \
  THCudaCheck(cudaPointerGetAttributes(&attributes,                     \
    torch::thc::data<ScalarType, THCTensorType>(state, input)));        \
  return torch::mpi::thc::broadcastp2pAsync<ScalarType, THCTensorType>( \
  state, input, root);                                                  \
}

#define DEFINE_NCCL_BROADCAST(ScalarType, THCTensorType)        \
  void PPCAT(torchmpi_nccl_broadcast_, THCTensorType)           \
    (THCState* state, THCTensorType *input, int root) {         \
  torch::nccl::thc::broadcast<ScalarType, THCTensorType>(       \
    state, input, root);                                        \
  }

#define DEFINE_NCCL_BROADCAST_ASYNC(ScalarType, THCTensorType)          \
  SynchronizationHandle* PPCAT(torchmpi_async_nccl_broadcast_, THCTensorType) \
    (THCState* state, THCTensorType *input, int root) {                 \
    return torch::nccl::thc::broadcastAsync<ScalarType, THCTensorType>( \
      state, input, root);                                              \
  }

/*********************** Reduce ************************************/
#define DEFINE_REDUCE(ScalarType, THCTensorType)                        \
  void PPCAT(torchmpi_reduce_, THCTensorType)                           \
    (THCState* state, THCTensorType *input, THCTensorType *output, int root) { \
    struct cudaPointerAttributes attributes;                            \
    THCudaCheck(cudaPointerGetAttributes(&attributes,                   \
      torch::thc::data<ScalarType, THCTensorType>(state, input)));      \
    torch::mpi::thc::reduce<ScalarType, THCTensorType>(                 \
      state, input, output, root, MPI_SUM);                             \
  }

#define DEFINE_NCCL_REDUCE(ScalarType, THCTensorType)                   \
  void PPCAT(torchmpi_nccl_reduce_, THCTensorType)                      \
    (THCState* state, THCTensorType *input, THCTensorType *output, int root) { \
  torch::nccl::thc::reduce<ScalarType, THCTensorType>(                  \
  state, input, output, root, ncclSum);                                 \
}

#define DEFINE_NCCL_REDUCE_ASYNC(ScalarType, THCTensorType)             \
  SynchronizationHandle* PPCAT(torchmpi_async_nccl_reduce_, THCTensorType) \
    (THCState* state, THCTensorType *input, THCTensorType *output, int root) { \
    return torch::nccl::thc::reduceAsync<ScalarType, THCTensorType>(    \
      state, input, output, root, ncclSum);                             \
  }

/*********************** Allreduce ************************************/
#define DEFINE_ALLREDUCE(ScalarType, THCTensorType)                     \
  void PPCAT(torchmpi_allreduce_, THCTensorType)(                       \
    THCState* state, THCTensorType *input, THCTensorType *output) {     \
    struct cudaPointerAttributes attributes;                            \
    THCudaCheck(cudaPointerGetAttributes(&attributes,                   \
      torch::thc::data<ScalarType, THCTensorType>(state, input)));      \
    torch::mpi::thc::allreduce<ScalarType, THCTensorType>(              \
      state, input, output, MPI_SUM);                                   \
  }

#define DEFINE_ALLREDUCE_ASYNC(ScalarType, THCTensorType)               \
  SynchronizationHandle*                                                \
  PPCAT(torchmpi_async_allreduce_, THCTensorType)(                      \
  THCState* state, THCTensorType *input, THCTensorType *output) {       \
  struct cudaPointerAttributes attributes;                              \
  THCudaCheck(cudaPointerGetAttributes(&attributes,                     \
    torch::thc::data<ScalarType, THCTensorType>(state, input)));        \
  return torch::mpi::thc::allreduceAsync<ScalarType, THCTensorType>(    \
    state, input, output, MPI_SUM);                                     \
}

#define DEFINE_ALLREDUCEP2P(ScalarType, THCTensorType)                  \
  void PPCAT(torchmpi_p2p_allreduce_, THCTensorType)(                   \
    THCState* state, THCTensorType *input, THCTensorType *output) {     \
    struct cudaPointerAttributes attributes;                            \
    THCudaCheck(cudaPointerGetAttributes(&attributes,                   \
      torch::thc::data<ScalarType, THCTensorType>(state, input)));      \
    torch::mpi::thc::allreducep2p<ScalarType, THCTensorType>(           \
      state, input, output, MPI_SUM);                                   \
  }

#define DEFINE_ALLREDUCEP2P_ASYNC(ScalarType, THCTensorType)            \
  SynchronizationHandle*                                                \
  PPCAT(torchmpi_async_p2p_allreduce_, THCTensorType)(                  \
    THCState* state, THCTensorType *input, THCTensorType *output) {     \
    struct cudaPointerAttributes attributes;                            \
    THCudaCheck(cudaPointerGetAttributes(&attributes,                   \
      torch::thc::data<ScalarType, THCTensorType>(state, input)));      \
    return torch::mpi::thc::allreducep2pAsync<ScalarType, THCTensorType>( \
      state, input, output, MPI_SUM);                                   \
  }

#define DEFINE_NCCL_ALLREDUCE(ScalarType, THCTensorType)                \
  void PPCAT(torchmpi_nccl_allreduce_, THCTensorType)(                  \
    THCState* state, THCTensorType *input, THCTensorType *output) {     \
    torch::nccl::thc::allreduce<ScalarType, THCTensorType>(             \
      state, input, output, ncclSum);                                   \
  }

#define DEFINE_NCCL_ALLREDUCE_ASYNC(ScalarType, THCTensorType)          \
  SynchronizationHandle* PPCAT(torchmpi_async_nccl_allreduce_, THCTensorType)( \
    THCState* state, THCTensorType *input, THCTensorType *output) {     \
    return torch::nccl::thc::allreduceAsync<ScalarType, THCTensorType>( \
      state, input, output, ncclSum);                                   \
  }

/*********************** Sendreceive **********************************/
#define DEFINE_SENDRECEIVE(ScalarType, THCTensorType)                   \
  void PPCAT(torchmpi_sendreceive_, THCTensorType)                      \
    (THCState* state, THCTensorType *input, int src, int dst) {         \
    struct cudaPointerAttributes attributes;                            \
    THCudaCheck(cudaPointerGetAttributes(&attributes,                   \
      torch::thc::data<ScalarType, THCTensorType>(state, input)));      \
    torch::mpi::thc::sendreceive<ScalarType, THCTensorType>(            \
      state, input, src, dst);                                          \
  }

/**********************************************************************
 ********************** C Wrapper instantiations **********************
 **********************************************************************/
#define FUNCTIONS_TO_INSTANTIATE_ALWAYS(                \
  CPP_TYPE, TH_TENSOR_TYPE, THC_TENSOR_TYPE)            \
  DEFINE_BROADCAST(CPP_TYPE, THC_TENSOR_TYPE);          \
  DEFINE_BROADCAST_ASYNC(CPP_TYPE, THC_TENSOR_TYPE);    \
  DEFINE_BROADCASTP2P(CPP_TYPE, THC_TENSOR_TYPE);       \
  DEFINE_BROADCASTP2P_ASYNC(CPP_TYPE, THC_TENSOR_TYPE); \
  DEFINE_REDUCE(CPP_TYPE, THC_TENSOR_TYPE);             \
  DEFINE_ALLREDUCE(CPP_TYPE, THC_TENSOR_TYPE);          \
  DEFINE_ALLREDUCE_ASYNC(CPP_TYPE, THC_TENSOR_TYPE);    \
  DEFINE_ALLREDUCEP2P(CPP_TYPE, THC_TENSOR_TYPE);       \
  DEFINE_ALLREDUCEP2P_ASYNC(CPP_TYPE, THC_TENSOR_TYPE); \
  DEFINE_SENDRECEIVE(CPP_TYPE, THC_TENSOR_TYPE);

#ifdef TORCH_MPI_NCCL
#define FUNCTIONS_TO_INSTANTIATE(                               \
  CPP_TYPE, TH_TENSOR_TYPE, THC_TENSOR_TYPE)                    \
  FUNCTIONS_TO_INSTANTIATE_ALWAYS(                              \
    CPP_TYPE, TH_TENSOR_TYPE, THC_TENSOR_TYPE);                 \
  DEFINE_NCCL_BROADCAST(CPP_TYPE, THC_TENSOR_TYPE);             \
  DEFINE_NCCL_BROADCAST_ASYNC(CPP_TYPE, THC_TENSOR_TYPE);       \
  DEFINE_NCCL_REDUCE(CPP_TYPE, THC_TENSOR_TYPE);                \
  DEFINE_NCCL_REDUCE_ASYNC(CPP_TYPE, THC_TENSOR_TYPE);          \
  DEFINE_NCCL_ALLREDUCE(CPP_TYPE, THC_TENSOR_TYPE);             \
  DEFINE_NCCL_ALLREDUCE_ASYNC(CPP_TYPE, THC_TENSOR_TYPE);
#else
#define FUNCTIONS_TO_INSTANTIATE(               \
  CPP_TYPE, TH_TENSOR_TYPE, THC_TENSOR_TYPE)    \
  FUNCTIONS_TO_INSTANTIATE_ALWAYS(              \
    CPP_TYPE, TH_TENSOR_TYPE, THC_TENSOR_TYPE);
#endif

#include "generic/torch_collectives_wrappers.cpp.in"

void torchmpi_free_ipc_descriptors() {
  VLOG_1("torchmpi_free_ipc_descriptors" << endl);
  auto& descs = getIPCDescs();
  descs = unordered_map<void*, unique_ptr<IPCDesc>> ();
}

} // extern "C"
