#ifndef TORCH_MPI_COLLECTIVES_INC
#define TORCH_MPI_COLLECTIVES_INC

#include "torch_mpi.h"

#include "TH.h"

namespace torch { namespace mpi {

// Collectives operating on scalar
template<typename T> void broadcastScalar(T& val, int src);

template<typename T> void allreduceScalar(T& val, MPI::Op mpiRedOp);

template<typename T> void sendreceiveScalar(T& t, int src, int dst);

namespace th { namespace detail {

  template<typename ScalarType> void allreducep2p(
    const ScalarType* inputData,
    ScalarType* outputData,
    size_t nElement,
    MPI::Op mpiRedOp,
    const MPI::Intracomm& comm);

}

// Collectives operating on TH*Tensor
template<typename ScalarType, typename THTensorType>
void broadcast(THTensorType* t, int src);

template<typename ScalarType, typename THTensorType>
resources::SynchronizationHandle* broadcastAsync(THTensorType* t, int src);

template<typename ScalarType, typename THTensorType>
void reduce(THTensorType* t,
            THTensorType* output,
            int dst,
            MPI::Op mpiRedOp);

template<typename ScalarType, typename THTensorType>
resources::SynchronizationHandle* reduceAsync(THTensorType* t,
                                                 THTensorType* output,
                                                 int dst,
                                                 MPI::Op mpiRedOp);

template<typename ScalarType, typename THTensorType>
void  allreduce(THTensorType* t,
                THTensorType* output,
                MPI::Op mpiRedOp);

template<typename ScalarType, typename THTensorType>
resources::SynchronizationHandle* allreduceAsync(THTensorType* t,
                                                 THTensorType* output,
                                                 MPI::Op mpiRedOp);

template<typename ScalarType, typename THTensorType>
void allreducep2p(THTensorType* t,
                  THTensorType* output,
                  MPI::Op mpiRedOp);

template<typename ScalarType, typename THTensorType>
resources::SynchronizationHandle* allreducep2pAsync(THTensorType* t,
                                                    THTensorType* output,
                                                    MPI::Op mpiRedOp);

template<typename ScalarType, typename THTensorType>
void sendreceive(THTensorType* t, int src, int dst);

}}}

#endif
