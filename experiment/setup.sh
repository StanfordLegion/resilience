#!/bin/bash

set -e

if [[ -z ${MACHINE} ]]; then
    echo "Did you remember to source experiments/MY_MACHINE_env.sh? (For an appropriate value of MY_MACHINE)"
    exit 1
fi

root_dir="$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"

git submodule update --init

if [[ ! -e legion ]]; then
    git clone -b regent-resilience-ppopp25 https://gitlab.com/StanfordLegion/legion.git
fi

if [[ $USE_HIP -eq 1 ]]; then
    if [[ ! -e Thrust ]]; then
        git clone https://github.com/ROCmSoftwarePlatform/Thrust.git
    fi
    export THRUST_PATH="$PWD/Thrust"
fi

pushd legion/language
# setup_env.py pins everything now, so don't need to pin explicitly here
DEBUG=0 CC=cc CXX=CC HOST_CC=gcc HOST_CXX=g++ USE_GASNET=1 REALM_NETWORKS=gasnetex ./scripts/setup_env.py --cmake --extra="-DCMAKE_INSTALL_PREFIX=$PWD/../install" --install -j${THREADS:-16}
popd

mkdir -p build
pushd build
resilience_flags=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_PREFIX_PATH=$PWD/../legion/install
    -DCMAKE_CXX_FLAGS="-Wall -DRESILIENCE_AUDIT_FUTURE_API"
    # do NOT set NDEBUG, it causes all sorts of issues
    -DCMAKE_CXX_FLAGS_RELEASE="-O2 -march=native"
    -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O2 -g -march=native"
)
cmake "${resilience_flags[@]}" ..
make -j${THREADS:-16}
popd
