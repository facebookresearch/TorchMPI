#! /bin/bash

set -ex

if ! command cat '1+1' | bc ; then
    echo "Need program bc to run this test"
    exit 1
fi

if ! test -e ./scripts/wrap.sh; then
    echo "Please run test from torchmpi base directory"
    exit 1
fi

LUAJIT=${LUAJIT:=luajit}

#########################################################################################################
# GPU tests
#########################################################################################################

# The following test suite is made for 2 nodes, 4 GPU per node which is the basic Nimbix setup.
# Single node tests
mpirun -n 4 --bind-to none ./scripts/wrap.sh ${LUAJIT} ./test/collectives.lua -all -gpu
mpirun -n 4 --bind-to none ./scripts/wrap.sh ${LUAJIT} ./test/collectives_nccl.lua -all -gpu
mpirun -n 2 --bind-to none ./scripts/wrap.sh ${LUAJIT} ./test/collectives_p2p.lua -all -gpu
mpirun -n 4 --bind-to none ./scripts/wrap.sh ${LUAJIT} ./test/collectives_p2p.lua -all -gpu
mpirun -n 4 --bind-to none ./scripts/wrap.sh ${LUAJIT} ./test/parameterserver.lua

# Longer tests, single node (apps)
mpirun -n 4 --bind-to none bash ./scripts/wrap.sh ${LUAJIT} ./apps/mnist/mnist_allreduce.lua -usegpu
mpirun -n 4 --bind-to none bash ./scripts/wrap.sh ${LUAJIT} ./apps/mnist/mnist_allreduce_async.lua -usegpu

# Parameterserver only CPU atm
# mpirun -n 4 --bind-to none bash ./scripts/wrap.sh ${LUAJIT} ./apps/mnist/mnist_parameterserver_dsgd.lua -usegpu
# mpirun -n 4 --bind-to none bash ./scripts/wrap.sh ${LUAJIT} ./apps/mnist/mnist_parameterserver_downpour.lua -usegpu
# mpirun -n 4 --bind-to none bash ./scripts/wrap.sh ${LUAJIT} ./apps/mnist/mnist_parameterserver_easgd.lua -usegpu

if test ${HOSTFILE}; then
    # No hostfile, no multi-node for you!
    stat ${HOSTFILE}

    # Multi-node tests
    mpirun -n 8 -hostfile ${HOSTFILE} --map-by node --bind-to none bash ./scripts/wrap.sh ${LUAJIT} ./test/collectives.lua -all -gpu

    # Custom hierarchical collectives have both cartesian and non-cartesian communicators, run a loop to test all
    export NUM_GPUS=$(nvidia-smi -L | wc -l)
    export NUM_NODES=$(cat ${HOSTFILE} | wc -l)
    export ub=$(echo ${NUM_GPUS}*${NUM_NODES} | bc)
    for n in $(seq 2 $ub); do
        mpirun -n ${n} -hostfile ${HOSTFILE} --map-by node --bind-to none bash ./scripts/wrap.sh ${LUAJIT} ./test/collectives_p2p.lua -all -gpu
        mpirun -n ${n} -hostfile ${HOSTFILE} --map-by node --bind-to none bash ./scripts/wrap.sh ${LUAJIT} ./test/collectives_nccl.lua -all -gpu
    done

    # Longer tests, multi-node (apps)
    mpirun -n 8 -hostfile ${HOSTFILE} --map-by node --bind-to none bash ./scripts/wrap.sh ${LUAJIT} ./apps/mnist/mnist_allreduce.lua -usegpu
    mpirun -n 8 -hostfile ${HOSTFILE} --map-by node --bind-to none bash ./scripts/wrap.sh ${LUAJIT} ./apps/mnist/mnist_allreduce_async.lua -usegpu

    # Parameterserver only CPU atm
    # mpirun -n 8 -hostfile ${HOSTFILE} --map-by node --bind-to none bash ./scripts/wrap.sh ${LUAJIT} ./apps/mnist/mnist_parameterserver_dsgd.lua -usegpu
    # mpirun -n 8 -hostfile ${HOSTFILE} --map-by node --bind-to none bash ./scripts/wrap.sh ${LUAJIT} ./apps/mnist/mnist_parameterserver_downpour.lua -usegpu
    # mpirun -n 8 -hostfile ${HOSTFILE} --map-by node --bind-to none bash ./scripts/wrap.sh ${LUAJIT} ./apps/mnist/mnist_parameterserver_easgd.lua -usegpu
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
