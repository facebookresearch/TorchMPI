#! /bin/bash

set -ex

if ! test -e ./scripts/wrap.sh; then
    echo "Please run test from torchmpi base directory"
    exit 1
fi

LUAJIT=${LUAJIT:=luajit}

#########################################################################################################
# CPU tests
#########################################################################################################
# For CPU only tests, we also pass --mca mpi_cuda_support 0
# Single node tests
mpirun -n 32 --bind-to none --mca mpi_cuda_support 0 ./scripts/wrap.sh ${LUAJIT} ./test/startstop.lua
mpirun -n 32 --bind-to none --mca mpi_cuda_support 0 ./scripts/wrap.sh ${LUAJIT} ./test/hierarchical_communicators.lua
mpirun -n 4  --bind-to none --mca mpi_cuda_support 0 ./scripts/wrap.sh ${LUAJIT} ./test/parameterserver.lua
mpirun -n 4  --bind-to none --mca mpi_cuda_support 0 ./scripts/wrap.sh ${LUAJIT} ./test/collectives.lua -all
mpirun -n 4  --bind-to none --mca mpi_cuda_support 0 ./scripts/wrap.sh ${LUAJIT} ./test/collectives_p2p.lua -all
mpirun -n 4  --bind-to none --mca mpi_cuda_support 0 ./scripts/wrap.sh ${LUAJIT} ./test/parameterserver.lua

# Longer tests, single node (examples)
${LUAJIT} examples/mnist/mnist_sequential.lua
mpirun -n 4  --bind-to none  --mca mpi_cuda_support 0 ./scripts/wrap.sh ${LUAJIT} ./examples/mnist/mnist_allreduce.lua
mpirun -n 4  --bind-to none  --mca mpi_cuda_support 0 ./scripts/wrap.sh ${LUAJIT} ./examples/mnist/mnist_allreduce_async.lua
mpirun -n 4  --bind-to none  --mca mpi_cuda_support 0 ./scripts/wrap.sh ${LUAJIT} ./examples/mnist/mnist_parameterserver_dsgd.lua
mpirun -n 4  --bind-to none  --mca mpi_cuda_support 0 ./scripts/wrap.sh ${LUAJIT} ./examples/mnist/mnist_parameterserver_downpour.lua
mpirun -n 4  --bind-to none  --mca mpi_cuda_support 0 ./scripts/wrap.sh ${LUAJIT} ./examples/mnist/mnist_parameterserver_easgd.lua

if test ${HOSTFILE}; then
    # No hostfile, no multi-node for you!
    stat ${HOSTFILE}

    # Multi-node tests
    mpirun -n 32 -hostfile ${HOSTFILE} --map-by node --bind-to none --mca mpi_cuda_support 0 ./scripts/wrap.sh ${LUAJIT} ./test/startstop.lua
    mpirun -n 32 -hostfile ${HOSTFILE} --map-by node --bind-to none  --mca mpi_cuda_support 0 ./scripts/wrap.sh ${LUAJIT} ./test/hierarchical_communicators.lua
    mpirun -n 8 -hostfile ${HOSTFILE} --map-by node --bind-to none  --mca mpi_cuda_support 0 ./scripts/wrap.sh ${LUAJIT} ./test/collectives.lua -all
    mpirun -n 8 -hostfile ${HOSTFILE} --map-by node --bind-to none  --mca mpi_cuda_support 0 ./scripts/wrap.sh ${LUAJIT} ./test/collectives_p2p.lua -all
    mpirun -n 8 -hostfile ${HOSTFILE} --map-by node --bind-to none  --mca mpi_cuda_support 0 ./scripts/wrap.sh ${LUAJIT} ./test/parameterserver.lua

    # Longer tests, multi-node (examples)
    mpirun -n 8 -hostfile ${HOSTFILE} --map-by node --bind-to none  --mca mpi_cuda_support 0 ./scripts/wrap.sh ${LUAJIT} ./examples/mnist/mnist_allreduce.lua
    mpirun -n 8 -hostfile ${HOSTFILE} --map-by node --bind-to none  --mca mpi_cuda_support 0 ./scripts/wrap.sh ${LUAJIT} ./examples/mnist/mnist_allreduce_async.lua
    mpirun -n 8 -hostfile ${HOSTFILE} --map-by node --bind-to none  --mca mpi_cuda_support 0 ./scripts/wrap.sh ${LUAJIT} ./examples/mnist/mnist_parameterserver_dsgd.lua
    mpirun -n 8 -hostfile ${HOSTFILE} --map-by node --bind-to none  --mca mpi_cuda_support 0 ./scripts/wrap.sh ${LUAJIT} ./examples/mnist/mnist_parameterserver_downpour.lua
    mpirun -n 8 -hostfile ${HOSTFILE} --map-by node --bind-to none  --mca mpi_cuda_support 0 ./scripts/wrap.sh ${LUAJIT} ./examples/mnist/mnist_parameterserver_easgd.lua
fi

# TODO: make this work properly in general
#cleanup() {
#    err=$?
#    echo "cleanup script"
#    pkill -9 terra && pkill -9 luajit
#    HOSTFILE=${HOSTFILE} . ./scripts/kill.sh
#    exit $err
#}
#trap cleanup SIGHUP SIGINT SIGTERM SIGEXIT
#
