#!/bin/bash

set -e
set -x

if [[ $INSTALL_DEPS -eq 1 ]]; then
    apt-get update -qq
    apt-get install -qq mpich libmpich-dev
fi

git submodule update --init

if [[ ! -e legion ]]; then
    git clone -b control_replication https://gitlab.com/StanfordLegion/legion.git
fi

pushd legion
if [[ ! -e build ]]; then
    mkdir build
    cd build
    legion_flags=(
        -DCMAKE_BUILD_TYPE=$([ ${DEBUG:-1} -eq 1 ] && echo Debug || echo Release)
        -DCMAKE_INSTALL_PREFIX=$PWD/../install
        -DCMAKE_CXX_STANDARD=11
        -DBUILD_SHARED_LIBS=ON # to improve link speed
    )
    if [[ ${DEBUG:-1} -eq 1 ]]; then
        legion_flags+=(
            -DLegion_BOUNDS_CHECKS=ON
            -DLegion_PRIVILEGE_CHECKS=ON
            -DBUILD_MARCH= # to avoid -march=native for valgrind compatibility
        )
    fi
    if [[ -n $LEGION_NETWORKS ]]; then
        legion_flags+=(
            -DLegion_NETWORKS=$LEGION_NETWORKS
        )
    fi
    cmake "${legion_flags[@]}" ..
    make install -j${THREADS:-4}
fi
popd

mkdir -p build
pushd build
resilience_flags=(
    -DCMAKE_BUILD_TYPE=$([ ${DEBUG:-1} -eq 1 ] && echo Debug || echo Release)
    -DCMAKE_PREFIX_PATH=$PWD/../legion/install
    -DCMAKE_CXX_FLAGS="-Wall -Werror -DRESILIENCE_AUDIT_FUTURE_API"
    # do NOT set NDEBUG, it causes all sorts of issues
    -DCMAKE_CXX_FLAGS_RELEASE="-O2 -march=native"
)
if [[ -n $LEGION_NETWORKS ]]; then
    resilience_flags+=(
        -DRESILIENCE_TEST_LAUNCHER="mpirun;-n;2"
    )
fi
cmake "${resilience_flags[@]}" ..
make -j${THREADS:-4}
REALM_SYNTHETIC_CORE_MAP= ctest --output-on-failure -j${THREADS:-4}
popd
