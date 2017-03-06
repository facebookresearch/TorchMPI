/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#include "../collectives.h"
#include "../resources.h"

#include <chrono>
#include <iostream>
#include <unordered_map>
#include <thread>
#include <vector>

/**********************************************************************
 ******************************** MPI *********************************
 **********************************************************************/
using namespace std;
using namespace torch::mpi::constants;
using namespace torch::mpi::resources;

namespace torch { namespace mpi { namespace th { namespace detail {

template<typename ScalarType> void reduce(
  ScalarType* out,
  const ScalarType* in,
  size_t size,
  decltype(MPI::Op(MPI_SUM)) mpiRedOp)
{
  for (size_t i = 0; i < size; ++i) {
    out[i] += in[i];
  }
}

// CPU memory hog, fine for now
void* getBuffer(void* dataPtr, int bufferSize, int bufferIndex = 0) {
  struct BufferWrapper {
    std::unordered_map<void*, std::vector<void*>> buffers_;
    ~BufferWrapper() {
      for (auto kvp : buffers_) {
        for (auto buf : kvp.second) {
          free(buf);
        }
      }
    }
  };
  static BufferWrapper wrap; // Wrap buffers into RAII layer
  static mutex mut;
  lock_guard<mutex> lg(mut);

  THAssert(bufferIndex < constants::kNumBuffersPerCollectiveCPU);
  auto& buffers = wrap.buffers_;
  if (buffers.find(dataPtr) == buffers.end()) {
    auto v = std::vector<void*>(constants::kNumBuffersPerCollectiveCPU);
    buffers.emplace(dataPtr, v);
  }
  THAssert(bufferIndex < constants::kNumBuffersPerCollectiveCPU);
  if (!buffers[dataPtr][bufferIndex]) {
    buffers[dataPtr][bufferIndex] = malloc(bufferSize);
  }
  return buffers[dataPtr][bufferIndex];
}

template<typename ScalarType>
void allreducep2p(
    const ScalarType* inputData,
    ScalarType* outputData,
    size_t nElement,
    MPI::Op mpiRedOp,
    const MPI::Intracomm& comm) {
  if (mpiRedOp != MPI::Op(MPI_SUM)) {
    THError("NYI: MPI_allreducep2p only supported for MPI_SUM atm");
  }

  auto rank = commRank(comm);
  auto size = commSize(comm);
  auto next = (rank + 1) % size;
  auto prev = (rank + size - 1) % size;

  auto bufferSize =
    std::max(
      static_cast<unsigned long>(1 << 13),
      (nElement + size * constants::kNumBuffersPerCollectiveCPU - 1) /
      (size * constants::kNumBuffersPerCollectiveCPU));
  auto rem = (nElement % bufferSize) ? 1 : 0;
  long totalChunks = nElement / bufferSize + rem;

  auto pp = getPlan<MpiPlan>(totalChunks, rank, rank, rank, size);
  auto& planReduce = pp.first;
  auto& planBroadcast = pp.second;
  // TODO: remove this extra copy
  if (outputData != inputData) {
    memcpy(outputData, inputData, sizeof(ScalarType) * nElement);
  }

  for (long step = 0; step < size - 1; ++step) {
    auto& plan = planReduce[step];
    std::vector<MPI::Request> reqSend(totalChunks), reqRecv(totalChunks);

    // 1. Post all IRecv to ensure receive buffer is allocated
    for (long startChunk = 0, startChunkIndex = 0; startChunk < totalChunks;
         startChunk += (long)size, ++startChunkIndex) {
      long sendingChunk = plan[startChunkIndex].first;
      long receivingChunk = plan[startChunkIndex].second;
      if (receivingChunk > -1) {
        auto buf = getBuffer(
          outputData, bufferSize * sizeof(ScalarType), startChunkIndex);
        auto receiveStart = receivingChunk * bufferSize;
        auto receiveEnd =
          std::min((receivingChunk + 1) * bufferSize - 1, nElement - 1);
        long len = (long)(receiveEnd - receiveStart + 1);

        if (receiveStart <= receiveEnd) {
          reqRecv[receivingChunk] = comm.Irecv(
            static_cast<ScalarType*>(buf),
            len,
            mpiType<ScalarType>(),
            prev,
            0);
        }
      }
    }

    // 2. Sync so that all buffers are allocated
    barrier(comm);

    // 3. Post all asynchronous ISend
    for (long startChunk = 0, startChunkIndex = 0; startChunk < totalChunks;
         startChunk += (long)size, ++startChunkIndex) {
      long sendingChunk = plan[startChunkIndex].first;
      long receivingChunk = plan[startChunkIndex].second;
      if (sendingChunk > -1) {
        auto sendStart = sendingChunk * bufferSize;
        auto sendEnd =
          std::min((sendingChunk + 1) * bufferSize - 1, nElement - 1);
        long len = (long)(sendEnd - sendStart + 1);

        if (sendStart <= sendEnd) {
          reqSend[sendingChunk] = comm.Isend(
            outputData + sendStart,
            len,
            mpiType<ScalarType>(),
            next,
            0);
        }
      }
    }

    // 4. Overlap compute and copies
    for (long startChunk = 0, startChunkIndex = 0; startChunk < totalChunks;
         startChunk += (long)size, ++startChunkIndex) {
      long sendingChunk = plan[startChunkIndex].first;
      long receivingChunk = plan[startChunkIndex].second;

      if (receivingChunk > -1) {
        auto buf = getBuffer(
          outputData, bufferSize * sizeof(ScalarType), startChunkIndex);
        reqRecv[receivingChunk].Wait();

        auto receiveStart = receivingChunk * bufferSize;
        auto receiveEnd =
          std::min((receivingChunk + 1) * bufferSize - 1, nElement - 1);
        long len = (long)(receiveEnd - receiveStart + 1);
        if (receiveStart <= receiveEnd) {
          reduce<ScalarType>(
            outputData + receiveStart,
            static_cast<ScalarType*>(buf),
            len,
            mpiRedOp
          );
        }
      }
    }

    // 5. Ensure all chunks are finished
    for (long startChunk = 0, startChunkIndex = 0; startChunk < totalChunks;
         startChunk += (long)size, ++startChunkIndex) {
      long sendingChunk = plan[startChunkIndex].first;
      long receivingChunk = plan[startChunkIndex].second;
      if (sendingChunk > -1) {
        reqSend[sendingChunk].Wait();
      }
    }
  }

  for (long step = 0; step < size - 1; ++step) {
    auto& plan = planBroadcast[step];
    std::vector<MPI::Request> reqSend(totalChunks), reqRecv(totalChunks);

    // 1. Post all IRecv to ensure receive buffer is allocated
    for (long startChunk = 0, startChunkIndex = 0; startChunk < totalChunks;
         startChunk += (long)size, ++startChunkIndex) {
      long sendingChunk = plan[startChunkIndex].first;
      long receivingChunk = plan[startChunkIndex].second;
      if (receivingChunk > -1) {
        auto receiveStart = receivingChunk * bufferSize;
        auto receiveEnd =
          std::min((receivingChunk + 1) * bufferSize - 1, nElement - 1);
        long len = (long)(receiveEnd - receiveStart + 1);

        if (receiveStart <= receiveEnd) {
          reqRecv[receivingChunk] = comm.Irecv(
            outputData + receiveStart,
            len,
            mpiType<ScalarType>(),
            prev,
            0);
        }
      }
    }

    // 2. Sync so that all buffers are allocated
    barrier(comm);

    // 3. Post all asynchronous ISend
    for (long startChunk = 0, startChunkIndex = 0; startChunk < totalChunks;
         startChunk += (long)size, ++startChunkIndex) {
      long sendingChunk = plan[startChunkIndex].first;
      long receivingChunk = plan[startChunkIndex].second;
      if (sendingChunk > -1) {
        auto sendStart = sendingChunk * bufferSize;
        auto sendEnd =
          std::min((sendingChunk + 1) * bufferSize - 1, nElement - 1);
        long len = (long)(sendEnd - sendStart + 1);
        if (sendStart <= sendEnd) {
          reqSend[sendingChunk] = comm.Isend(
            outputData + sendStart,
            len,
            mpiType<ScalarType>(),
            next,
            0);
        }
      }
    }

    // 4. Ensure all chunks are finished
    for (auto& r : reqSend) { r.Wait(); }
    for (auto& r : reqRecv) { r.Wait(); }
  }
}

}}}} // ns torch::mpi::th::detail

// Explicit template instantiations
template void
torch::mpi::th::detail::allreducep2p<uint8_t>(
  const uint8_t*, uint8_t*, size_t nElement, decltype(MPI::Op(MPI_SUM)), const MPI::Intracomm& comm);
template void
torch::mpi::th::detail::allreducep2p<char>(
  const char*, char*, size_t nElement, decltype(MPI::Op(MPI_SUM)), const MPI::Intracomm& comm);
template void
torch::mpi::th::detail::allreducep2p<short>(
  const short*, short*, size_t nElement, decltype(MPI::Op(MPI_SUM)), const MPI::Intracomm& comm);
template void
torch::mpi::th::detail::allreducep2p<int>(
  const int*, int*, size_t nElement, decltype(MPI::Op(MPI_SUM)), const MPI::Intracomm& comm);
template void
torch::mpi::th::detail::allreducep2p<long>(
  const long*, long*, size_t nElement, decltype(MPI::Op(MPI_SUM)), const MPI::Intracomm& comm);
template void
torch::mpi::th::detail::allreducep2p<float>(
  const float*, float*, size_t nElement, decltype(MPI::Op(MPI_SUM)), const MPI::Intracomm& comm);
template void
torch::mpi::th::detail::allreducep2p<double>(
  const double*, double*, size_t nElement, decltype(MPI::Op(MPI_SUM)), const MPI::Intracomm& comm);
