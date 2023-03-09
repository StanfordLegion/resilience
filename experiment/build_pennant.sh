#!/bin/bash

set -e

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

export INCLUDE_PATH="$root_dir/../src"
export LIB_PATH="$root_dir/../build/src"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$root_dir/../build/src"

mkdir "$1"
cd "$1"

SAVEOBJ=1 STANDALONE=1 OBJNAME=./pennant.checkpoint $root_dir/../legion/language/regent.py $root_dir/../legion/language/examples/pennant.rg -fpredicate 0 -fflow 0 -fopenmp 0 -fcuda 1 -fcuda-offline 1 -fcuda-arch pascal -fcuda-generate-cubin 1 -fcheckpoint 1

cp $root_dir/../build/src/liblegion_resilience.so .

cp $root_dir/*_pennant*.sh .
cp -r $root_dir/pennant.tests .
