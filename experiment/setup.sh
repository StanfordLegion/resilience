#!/bin/bash

set -e

root_dir="$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"

git submodule update --init

if [[ ! -e legion ]]; then
    git clone -b regent-resilience https://gitlab.com/StanfordLegion/legion.git
fi

pushd legion/language
# setup_env.py pins everything now, so don't need to pin explicitly here
DEBUG=0 CC=cc CXX=CC HOST_CC=gcc HOST_CXX=g++ USE_GASNET=1 REALM_NETWORKS=gasnetex USE_CUDA=1 ./scripts/setup_env.py --cmake --extra='-DCMAKE_INSTALL_PREFIX=$PWD/../install' --install
popd

mkdir -p build
pushd build
resilience_flags=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_PREFIX_PATH=$PWD/../legion/install
    -DCMAKE_CXX_FLAGS="-Wall -DRESILIENCE_AUDIT_FUTURE_API"
    # do NOT set NDEBUG, it causes all sorts of issues
    -DCMAKE_CXX_FLAGS_RELEASE="-O2 -march=native"
)
cmake "${resilience_flags[@]}" ..
make -j${THREADS:-16}
popd
