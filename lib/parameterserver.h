#pragma once

namespace torch { namespace mpi {

std::thread& parameterServerThread();
void freeParameterServers();

}} // ns torch.mpi
