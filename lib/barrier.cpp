#include "resources.h"
using namespace torch::mpi::resources;

extern "C" void customBarrier() {
  auto s = Barrier::acquire();
  s->barrier();
  Barrier::release(s);
}
