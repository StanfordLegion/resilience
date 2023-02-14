#!/bin/bash

set -e
set -x

git submodule update --init

if [[ ! -e legion ]]; then
    git clone -b control_replication https://gitlab.com/StanfordLegion/legion.git
fi

pushd legion
if [[ ! -e build ]]; then
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$PWD/../install -DCMAKE_CXX_STANDARD=11 ..
    make install -j${THREADS:-4}
fi
popd

mkdir -p build
pushd build
cmake -DCMAKE_PREFIX_PATH=$PWD/../legion/install ..
make -j${THREADS:-4}
ctest -j${THREADS:-4}
popd
