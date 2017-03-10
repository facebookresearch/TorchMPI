#!/bin/sh

set -e

# ./scripts/ompirun.sh -n 4 -npernode 4 -hostfile ${HOSTFILE} ./scripts/wrap.sh ${LUAJIT} ./test/collectives_all.lua -storage inplace -execution sync -processor gpu -tests p2p -benchmark -hierarchical false -staged false -numBuffers 2

OMPI_OPTIONS=${OMPI_OPTIONS:='--mca btl_smcuda_cuda_max_send_size 10000000 --mca btl_smcuda_rndv_eager_limit 10000000 --mca btl_smcuda_eager_limit 10000000 --mca btl_smcuda_max_send_size 10000000 --mca btl_openib_max_send_size 10000000 --bind-to none'}

mpirun ${OMPI_OPTIONS} $@
