# Luarocks and Torch
```
git clone https://github.com/torch/luajit-rocks.git
cd luajit-rocks
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/your/prefix
make install
```

# Install Torch, cuTorch, cuDNN wrappers
```
/your/prefix/bin/luarocks install torchnet
/your/prefix/bin/luarocks install cutorch
/your/prefix/bin/luarocks install cunn
/your/prefix/bin/luarocks install cudnn
```

# NCCL
```
git clone https://github.com/NVIDIA/nccl.git && cd nccl && make -j 16 && make install
```

# OpenMPI
```
wget https://www.open-mpi.org/software/ompi/v2.0/downloads/openmpi-2.0.1.tar.bz2 && bunzip2 openmpi-2.0.1.tar.bz2 && cd openmpi
./configure --prefix=/usr/local --enable-mpi-cxx --enable-shared --with-slurm --enable-event-thread-support --enable-opal-multi-threads --enable-orte-progress-threads --enable-mpi-thread-multiple --enable-mpi-ext=affinity,cuda --with-cuda=/usr/local/cuda
```

# Deploy on Nimbix
```
head -n 1 /etc/JARVICE/nodes | xargs -i ssh {} "cd /data/nccl && make install" && cat /etc/JARVICE/nodes | xargs -n 1 -P 1 -i ssh {} "cd /data/nccl && make install"

head -n 1 /etc/JARVICE/nodes | xargs -i ssh {} "cd /data/torchmpi && MPI_C_COMPILER=$(which mpicc) MPI_CXX_COMPILER=$(which mpic++) /usr/local/bin/luarocks make rocks/torch_mpi-scm-1.rockspec" && cat /etc/JARVICE/nodes | xargs -n 1 -P 1 -i ssh {} "cd /data/torchmpi && MPI_C_COMPILER=$(which mpicc) MPI_CXX_COMPILER=$(which mpic++) /usr/local/bin/luarocks make rocks/torch_mpi-scm-1.rockspec"
```

# Killall on Nimbix
```
cat /etc/JARVICE/nodes | xargs -n 1 -P 8 -i ssh {} "pkill -9 terra && pkill -9 luajit"
```
