# TorchMPI

## Local install (tested on MacOS and Ubuntu @ DockerHub: nicolasvasilache/torchmpi-devel)

Please first check dependencies you need:
  - [CUDA Toolkit](https://developer.nvidia.com/cuda-zone) if you need CUDA
  - [cutorch](https://github.com/torch/cutorch) also mandatory if you need CUDA
  - [NCCL](https://github.com/NVIDIA/nccl) for faster collectives with CUDA (optional)

If CUDA and cutorch are not found, the installation will install the
CPU-only version.

Once you are ready, just run the following `luarocks` command:
```sh
MPI_C_COMPILER=<path to>/mpicc MPI_CXX_COMPILER=<path to>/mpicxx MPI_CXX_COMPILE_FLAGS="-O3" <path to>/luarocks make rocks/torch_mpi-scm-1.rockspec
```
Note that on certain system your MPI compilers might have different
names. `MPI_C_COMPILER` and `MPI_CXX_COMPILER` are optional (the install
will try to find MPI), but if MPI is not found, _both_ must be specified.
