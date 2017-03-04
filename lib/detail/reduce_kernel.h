#pragma once

#include <cuda_runtime.h>

namespace torch { namespace mpi { namespace thc { namespace detail {

template<typename ScalarType>
void reduce(ScalarType* out, const ScalarType* in, unsigned long size, cudaStream_t stream);

}}}}
