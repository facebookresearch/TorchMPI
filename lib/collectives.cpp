#include "collectives.h"

#include <chrono>
#include <iostream>
#include <unordered_map>
#include <thread>
#include <vector>

#include "resources.h"

/**********************************************************************
 ******************************** MPI *********************************
 **********************************************************************/
using namespace std;
using namespace torch::th;
using namespace torch::mpi::constants;
using namespace torch::mpi::resources;

namespace torch { namespace mpi {

  //////////////////////////////////////////////////////////////////////////////
  // Collectives operating on scalars, this is latency-bound.
  /////////////////////////////////////////////////////////////////////////////
  template<typename T>
  void broadcastScalar(T& val, int src) {
    mpi::getMainThreadCommunicator().intraComm.Bcast(&val, 1, mpiType<T>(), src);
  }

  template<typename T>
  void reduceScalar(T& val, int dst) {
    mpi::getMainThreadCommunicator().intraComm.Reduce(
      MPI_IN_PLACE, &val, 1, mpiType<T>(), dst);
  }

  template<typename T> void allreduceScalar(
      T& val, MPI::Op mpiRedOp) {
    mpi::getMainThreadCommunicator().intraComm.Allreduce(
      MPI_IN_PLACE, &val, 1, mpiType<T>(), mpiRedOp);
  }

  template<typename T> void sendreceiveScalar(
      T& input, int src, int dst) {
    mpi::getMainThreadCommunicator().intraComm.Sendrecv_replace(
      &input, 1, mpiType<T>(), dst, kDefaultTag, src, kDefaultTag);
  }

namespace th {

#define PREPARE(tensor)                                                 \
  if (!isContiguous(tensor)) {                                          \
    THError("NYI: MPI_Sendrecv_replace only supported for contig tensors"); \
  }                                                                     \
  retainStorage(tensor);                                                \
  auto tensorData = data<ScalarType>(tensor);                           \
  auto nElement = torch::th::nElement<THTensorType>(tensor);            \
  auto collectiveLevel = torch::mpi::getCollectiveSpan().first;         \
  CommunicatorGuard cs(collectiveLevel);                                \
  const CollectiveResources* r = acquireCollectiveResources(            \
    tensorData, Spin(true));

#define PREPARE2(input, output)                                        \
  if (!isContiguous(input)) {                                           \
    THError("NYI: MPI_Bcast only supported for contig tensors");        \
  }                                                                     \
                                                                        \
  retainStorage(input);                                                 \
  if (input != output) {                                                \
    retainStorage(output);                                              \
  }                                                                     \
  auto inputData = data<ScalarType>(input);                             \
  auto outputData = (output) ? data<ScalarType>(output) : inputData;    \
  auto nElement = torch::th::nElement<THTensorType>(input);             \
  auto collectiveLevel = torch::mpi::getCollectiveSpan().first;         \
  CommunicatorGuard cs(collectiveLevel);                                \
  const CollectiveResources* r = acquireCollectiveResources(            \
    inputData, Spin(true));

  //////////////////////////////////////////////////////////////////////
  // Collectives operating on TH*Tensor
  //////////////////////////////////////////////////////////////////////

  template<typename ScalarType, typename THTensorType>
  void sendreceive(THTensorType* tensor,
                   int src,
                   int dst) {
    PREPARE(tensor);
    r->comm->intraComm.Sendrecv_replace(tensorData,
                              nElement,
                              mpiType<ScalarType>(),
                              dst,
                              kDefaultTag,
                              src,
                              kDefaultTag);
    // TODO: ScopeGuard
    releaseCollectiveResources(const_cast<CollectiveResources*>(r));
  }

  template<typename ScalarType, typename THTensorType>
  void broadcast(THTensorType* tensor, int src) {
    PREPARE(tensor);
    r->comm->intraComm.Bcast(tensorData, nElement, mpiType<ScalarType>(), src);
    // TODO: ScopeGuard
    releaseCollectiveResources(const_cast<CollectiveResources*>(r));
  }

  template<typename ScalarType, typename THTensorType>
  void reduce(THTensorType* input,
              THTensorType* output,
              int dst,
              MPI::Op mpiRedOp)
  {
    PREPARE2(input, output);

    if (outputData == inputData) {
      r->comm->intraComm.Reduce(
        (commRank(r->comm->intraComm) == dst) ? MPI_IN_PLACE : inputData,
        (commRank(r->comm->intraComm) == dst) ? outputData : nullptr,
        nElement,
        mpiType<ScalarType>(),
        mpiRedOp,
        dst);
    } else {
      r->comm->intraComm.Reduce(
        inputData,
        outputData,
        nElement,
        mpiType<ScalarType>(),
        mpiRedOp,
        dst);
    }

    // TODO: ScopeGuard
    releaseCollectiveResources(const_cast<CollectiveResources*>(r));
  }

  template<typename ScalarType>
  void allreduceImpl(ScalarType* input,
                     ScalarType* output,
                     size_t nElement,
                     MPI::Op mpiRedOp,
                     const CollectiveResources* r) {
    r->comm->intraComm.Allreduce(
      (output != input) ? input : MPI_IN_PLACE,
      output,
      nElement,
      mpiType<ScalarType>(),
      mpiRedOp);
  }

  template<typename ScalarType, typename THTensorType>
  void allreduce(THTensorType* input,
                 THTensorType* output,
                 MPI::Op mpiRedOp) {
    PREPARE2(input, output);

    allreduceImpl(inputData, outputData, nElement, mpiRedOp, r);

    // TODO: ScopeGuard
    releaseCollectiveResources(const_cast<CollectiveResources*>(r));
  }

  template<typename ScalarType, typename THTensorType>
  void allreducep2p(THTensorType* input,
                    THTensorType* output,
                    MPI::Op mpiRedOp) {
    PREPARE2(input, output);

    if (nElement <= constants::kSmallAllreduceSizeCPU) {
      // Go through CPU
      allreduceImpl(inputData, outputData, nElement, mpiRedOp, r);
    } else {
      detail::allreducep2p<ScalarType>(
        inputData, outputData, nElement, mpiRedOp, r->comm->intraComm);
    }

    // TODO: ScopeGuard
    releaseCollectiveResources(const_cast<CollectiveResources*>(r));
  }

  /* Asynchronous CPU-side collectives */
  template<typename ScalarType, typename THTensorType>
  SynchronizationHandle* broadcastAsync(THTensorType* tensor, int src)
  {
    PREPARE(tensor);

    MPI_Request req;
    MPI_Ibcast(tensorData,
               nElement,
               mpiType<ScalarType>(),
               src,
               r->comm->intraComm,
               &req);

    // TODO: ScopeGuard
    releaseCollectiveResources(const_cast<CollectiveResources*>(r));

    return resources::synchronizationHandleFromMPIRequest(
      enqueueMPIRequest(MPI::Request(req)));
  }

  template<typename ScalarType, typename THTensorType>
  SynchronizationHandle* reduceAsync(THTensorType* input,
                                     THTensorType* output,
                                     int dst,
                                     MPI::Op mpiRedOp)
  {
    PREPARE2(input, output);

    MPI_Request req;
    if (outputData == inputData) {
      MPI_Ireduce(
        (commRank(r->comm->intraComm) == dst) ? MPI_IN_PLACE : inputData,
        (commRank(r->comm->intraComm) == dst) ? outputData : nullptr,
        nElement,
        mpiType<ScalarType>(),
        mpiRedOp,
        dst,
        r->comm->intraComm,
        &req);
    } else {
      MPI_Ireduce(
        inputData,
        outputData,
        nElement,
        mpiType<ScalarType>(),
        mpiRedOp,
        dst,
        r->comm->intraComm,
        &req);
    }

    // TODO: ScopeGuard
    releaseCollectiveResources(const_cast<CollectiveResources*>(r));

    return resources::synchronizationHandleFromMPIRequest(
      enqueueMPIRequest(MPI::Request(req)));
  }

  template<typename ScalarType, typename THTensorType>
  SynchronizationHandle* allreduceAsync(THTensorType* input,
                                        THTensorType* output,
                                        MPI::Op mpiRedOp)
  {
    PREPARE2(input, output);

    MPI_Request req;
    MPI_Iallreduce(
      (outputData != inputData) ? inputData : MPI_IN_PLACE,
      outputData,
      nElement,
      mpiType<ScalarType>(),
      mpiRedOp,
      r->comm->intraComm,
      &req);

    // TODO: ScopeGuard
    releaseCollectiveResources(const_cast<CollectiveResources*>(r));

    return resources::synchronizationHandleFromMPIRequest(
      enqueueMPIRequest(MPI::Request(req)));
  }

  template<typename ScalarType, typename THTensorType>
  resources::SynchronizationHandle*  allreducep2pAsync(
    THTensorType* input,
    THTensorType* output,
    MPI::Op mpiRedOp)
  {
    PREPARE2(input, output);

    auto& futures = getCollectiveFutures();
    futures.push_back(
      collectiveOffloadThreadPool().enqueue([=](){
        if (nElement <= constants::kSmallAllreduceSizeCPU) {
          // Go through CPU
          allreduceImpl(inputData, outputData, nElement, mpiRedOp, r);
        } else {
          detail::allreducep2p<ScalarType>(
            inputData, outputData, nElement, mpiRedOp, r->comm->intraComm);
        }
      // TODO: ScopeGuard??
      releaseCollectiveResources(const_cast<CollectiveResources*>(r));
    }));

    return resources::synchronizationHandleFromFuture(futures.size() - 1);
  }

}}} // ns torch::mpi::th


// Explicit template instantiations

/**********************************************************************
 *********************** C Wrapper definitions ************************
 **********************************************************************/
#define PPCAT_NX(A, B) A ## B
#define PPCAT(A, B) PPCAT_NX(A, B)

extern "C" {

/*********************** Broadcast ************************************/
#define DEFINE_BROADCAST_SCALAR(ElemType)      \
  ElemType PPCAT(torchmpi_broadcast_, ElemType) (       \
    ElemType val, int src) {                            \
    torch::mpi::broadcastScalar(val, src);     \
    return val;                                         \
  }

#define DEFINE_BROADCAST(ScalarType, THTensorType)                \
  void                                                                     \
  PPCAT(torchmpi_broadcast_, THTensorType)(THTensorType *input, int src) { \
    torch::mpi::th::broadcast<ScalarType, THTensorType>(                   \
      input, src);                                                \
  }

#define DEFINE_BROADCAST_ASYNC(ScalarType, THTensorType)       \
  SynchronizationHandle*                                                \
  PPCAT(torchmpi_async_broadcast_, THTensorType)(THTensorType *input,   \
                                                 int src)               \
  {                                                                     \
    return torch::mpi::th::broadcastAsync<ScalarType, THTensorType>(    \
      input, src);                                             \
  }

/*********************** Reduce ************************************/
#define DEFINE_REDUCE_SCALAR(ElemType)                 \
  ElemType PPCAT(torchmpi_reduce_, ElemType) (                  \
    ElemType val, int dst) {                                    \
    torch::mpi::reduceScalar(val, dst, MPI_SUM);       \
    return val;                                                 \
  }

#define DEFINE_REDUCE(ScalarType, THTensorType)        \
  void PPCAT(torchmpi_reduce_, THTensorType)(                   \
    THTensorType *input, THTensorType *output, int dst) {       \
    torch::mpi::th::reduce<ScalarType, THTensorType>(           \
      input, output, dst, MPI_SUM);                    \
  }

#define DEFINE_REDUCE_ASYNC(ScalarType, THTensorType)          \
  SynchronizationHandle*                                                \
  PPCAT(torchmpi_async_reduce_, THTensorType)(THTensorType *input,      \
                                              THTensorType *output,     \
                                              int dst)                  \
  {                                                                     \
    return torch::mpi::th::reduceAsync<ScalarType, THTensorType>(       \
      input, output, dst, MPI_SUM);                            \
  }

/*********************** Allreduce ************************************/
#define DEFINE_ALLREDUCE_SCALAR(ElemType)                      \
  ElemType PPCAT(torchmpi_allreduce_, ElemType) (ElemType val) {        \
    torch::mpi::allreduceScalar(val, MPI_SUM);                 \
    return val;                                                         \
  }

#define DEFINE_ALLREDUCE(ScalarType, THTensorType)     \
  void PPCAT(torchmpi_allreduce_, THTensorType)(       \
    THTensorType *input, THTensorType *output) {                \
    torch::mpi::th::allreduce<ScalarType, THTensorType>(        \
      input, output, MPI_SUM);                         \
  }

#define DEFINE_ALLREDUCE_ASYNC(ScalarType, THTensorType)       \
  SynchronizationHandle* PPCAT(torchmpi_async_allreduce_, THTensorType)( \
    THTensorType *input, THTensorType *output) {                        \
    return torch::mpi::th::allreduceAsync<ScalarType, THTensorType>(    \
      input, output, MPI_SUM);                                 \
  }

#define DEFINE_ALLREDUCEP2P(ScalarType, THTensorType)  \
  void PPCAT(torchmpi_p2p_allreduce_, THTensorType)(   \
    THTensorType *input, THTensorType *output) {                \
    torch::mpi::th::allreducep2p<ScalarType, THTensorType>(     \
      input, output, MPI_SUM);                         \
  }

#define DEFINE_ALLREDUCEP2P_ASYNC(ScalarType, THTensorType)    \
  void PPCAT(torchmpi_async_p2p_allreduce_, THTensorType)(               \
    THTensorType *input, THTensorType *output) {                        \
    torch::mpi::th::allreducep2pAsync<ScalarType, THTensorType>(        \
      input, output, MPI_SUM);                                 \
  }

/*********************** Sendreceive **********************************/
#define DEFINE_SENDRECEIVE_SCALAR(ElemType)            \
  ElemType PPCAT(torchmpi_sendreceive_, ElemType) (    \
    ElemType val, int src, int dst) {                           \
    torch::mpi::sendreceiveScalar(val, src, dst);      \
    return val;                                                 \
  }

#define DEFINE_SENDRECEIVE(ScalarType, THTensorType)   \
  void PPCAT(torchmpi_sendreceive_, THTensorType) (    \
    THTensorType *input, int src, int dst) {                    \
    torch::mpi::th::sendreceive<ScalarType, THTensorType>(      \
      input, src, dst);                                \
  }

/**********************************************************************
 ********************** C Wrapper instantiations **********************
 **********************************************************************/
#define FUNCTIONS_TO_INSTANTIATE(CPP_TYPE, TH_TENSOR_TYPE, THC_TENSOR_TYPE) \
  DEFINE_BROADCAST_SCALAR(CPP_TYPE);                                    \
  DEFINE_ALLREDUCE_SCALAR(CPP_TYPE);                                    \
  DEFINE_SENDRECEIVE_SCALAR(CPP_TYPE);                                  \
                                                                        \
  DEFINE_BROADCAST(CPP_TYPE, TH_TENSOR_TYPE);                           \
  DEFINE_REDUCE(CPP_TYPE, TH_TENSOR_TYPE);                              \
  DEFINE_ALLREDUCE(CPP_TYPE, TH_TENSOR_TYPE);                           \
  DEFINE_ALLREDUCEP2P(CPP_TYPE, TH_TENSOR_TYPE);                        \
  DEFINE_SENDRECEIVE(CPP_TYPE, TH_TENSOR_TYPE);                         \
                                                                        \
  DEFINE_BROADCAST_ASYNC(CPP_TYPE, TH_TENSOR_TYPE);                     \
  DEFINE_REDUCE_ASYNC(CPP_TYPE, TH_TENSOR_TYPE);                        \
  DEFINE_ALLREDUCEP2P_ASYNC(CPP_TYPE, TH_TENSOR_TYPE);                  \
  DEFINE_ALLREDUCE_ASYNC(CPP_TYPE, TH_TENSOR_TYPE);

#include "generic/torch_collectives_wrappers.cpp.in"


} // extern "C"
