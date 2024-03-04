#!/bin/bash

set -e
set -x

if [[ $INSTALL_DEPS -eq 1 ]]; then
    $SUDO_COMMAND apt-get update -qq
    $SUDO_COMMAND apt-get install -qq mpich libmpich-dev
    if [[ ${USE_REGENT:-0} -eq 1 ]]; then
        $SUDO_COMMAND apt-get install -qq llvm-11-dev clang-11 libclang-11-dev libedit-dev libncurses5-dev libffi-dev libpfm4-dev libxml2-dev
        export CMAKE_PREFIX_PATH=/usr/lib/llvm-11:/usr/share/llvm-11
    fi
fi

git submodule update --init

if [[ ! -e legion ]]; then
    git clone -b regent-resilience-sc24 https://gitlab.com/StanfordLegion/legion.git
fi

pushd legion
if [[ ! -e build ]]; then
    mkdir build
    pushd build
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
    if [[ ${USE_REGENT:-0} -eq 1 ]]; then
        legion_flags+=(
            -DLegion_BUILD_BINDINGS=ON
        )
    fi
    if [[ -n $LEGION_NETWORKS ]]; then
        legion_flags+=(
            -DLegion_NETWORKS=$LEGION_NETWORKS
        )
    fi
    cmake "${legion_flags[@]}" ..
    make install -j${THREADS:-4}
    popd
    if [[ ${USE_REGENT:-0} -eq 1 ]]; then
        pushd language
        ./install.py --legion-install-prefix=$PWD/../install --rdir=auto
        popd
    fi
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
    -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O2 -g -march=native"
)
if [[ -n $LEGION_NETWORKS ]]; then
    resilience_flags+=(
        -DRESILIENCE_TEST_LAUNCHER="mpirun;-n;2"
    )
fi
cmake "${resilience_flags[@]}" ..
make -j${THREADS:-4}
export REALM_SYNTHETIC_CORE_MAP=
if [[ ${USE_REGENT:-0} -eq 1 ]]; then
    (
        export INCLUDE_PATH="$PWD/../src"
        export LIB_PATH="$PWD/src"
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD/src"
        cd ../legion/language
        ./test.py -j${THREADS:-4}
    )
else
    ctest --output-on-failure -j${THREADS:-4}
fi
popd
