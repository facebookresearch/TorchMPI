#pragma once

#include <mpi.h>

#ifdef TORCH_MPI_NCCL
#include "nccl.h"
#endif

namespace torch { namespace mpi { namespace constants {

template<typename T> MPI::Datatype mpiType();

#ifdef TORCH_MPI_NCCL
template<typename T> ncclDataType_t ncclType();
MPI::Op mpiOp(ncclRedOp_t ncclRedOp);
#endif

extern bool immutableConstants;

extern bool kUseStagedCollectives;
extern bool kUseHierarchicalCollectives;

extern int kCollectiveOffloadThreadPoolSize;
extern int kNumAsyncCollectivesInFlight;
extern int kParameterServerOffloadThreadPoolSize;
extern int kNumAsyncParameterServersInFlight;

extern int kSmallBcastSizeCPU;
extern int kSmallAllreduceSizeCPU;
extern int kSmallBcastSizeGPU;
extern int kSmallAllreduceSizeGPU;

// Min transfer size to reaming in a nice latency regime (131K matches the
// default OpenMPI transfer size)
extern int kMinGPUBufferSize;
// 4MB max per buffer per thread
extern int kMaxGPUBufferSize;

// Min transfer size to reaming in a nice latency regime (131K matches the
// default OpenMPI transfer size)
extern int kMinCPUBufferSize;
// 4MB max per buffer per thread
extern int kMaxCPUBufferSize;

extern int kNumBuffersPerCollectiveCPU;
extern int kNumBuffersPerCollectiveGPU;

extern int kCollectiveOffloadThreadPoolSize;
extern int kNumAsyncCollectivesInFlight;
extern int kParameterServerOffloadThreadPoolSize;
extern int kNumAsyncParameterServersInFlight;

constexpr int kNumAsyncRequestsInFlight = 1 << 20;

// These are used to disambiguate MPI_Send / MPI_Recv tags
constexpr long kMaxNumBuffersPerCollectiveCPU = 16;
constexpr long kMaxNumBuffersPerCollectiveGPU = 16;

}}} // torch::mpi::constants
