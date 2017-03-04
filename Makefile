# For Nimbix IBM/Minsky machines
LUAROCKS_INSTALL_DIR=/opt/DL/torch/
all:
	mpicxx -std=c++11 -I${LUAROCKS_INSTALL_DIR}/include/TH -I${LUAROCKS_INSTALL_DIR}/include/THC -I/usr/local/cuda/include -DTORCH_MPI_CUDA=1 lib/torch_mpi.cpp
	mpicxx -std=c++11 -I${LUAROCKS_INSTALL_DIR}/include/TH -I${LUAROCKS_INSTALL_DIR}/include/THC -I/usr/local/cuda/include -DTORCH_MPI_CUDA=1 lib/collectives.cpp
	mpicxx -std=c++11 -I${LUAROCKS_INSTALL_DIR}/include/TH -I${LUAROCKS_INSTALL_DIR}/include/THC -I/usr/local/cuda/include -DTORCH_MPI_CUDA=1 lib/parameterserver.cpp
