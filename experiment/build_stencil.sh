#!/bin/bash

set -e

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

export INCLUDE_PATH="$root_dir/../src"
export LIB_PATH="$root_dir/../build/src"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$root_dir/../build/src"

mkdir "$1"
cd "$1"

USE_FOREIGN=0 SAVEOBJ=1 STANDALONE=1 OBJNAME=./stencil.checkpoint $root_dir/../legion/language/regent.py $root_dir/../legion/language/examples/stencil_fast.rg -fpredicate 0 -fflow 0 -fopenmp 0 -foverride-demand-cuda 1 $REGENT_GPU_FLAGS -fcheckpoint 1

cp $root_dir/../build/src/liblegion_resilience.so .

cp $root_dir/$MACHINE/*_stencil*.sh .
