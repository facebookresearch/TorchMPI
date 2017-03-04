# MNIST
The mnist dataset is installed automatically via luarocks in torch/install.
CPU training on 8 MPI processes, 1 per GPU (with overprovisionning round robin if necessary)
```
mpirun --bind-to none -n 8 luajit ../apps/mnist/mnist_allreduce.lua
```
GPU training on 8 MPI processes, 1 per GPU (with overprovisionning round robin if necessary)
```
mpirun --mca btl_smcuda_use_cuda_ipc 1 --bind-to none -n 8 luajit ../apps/mnist/mnist_allreduce.lua -usegpu
```

# Imagenet with resnet
Pull in the resnet files
```
git submodule init && git submodule update
```
Download and extract imagenenet dataset in /data/imagenet
```
CUDA_VISIBLE_DEVICES=0 luajit ./resnet/main.lua -data /data/imagenet
```
