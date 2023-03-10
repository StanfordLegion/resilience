#!/bin/bash

set -e

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

mkdir "$1"
cd "$1"

cp $root_dir/../legion/install/lib*/*.so* .
cp $root_dir/../build/examples/circuit/circuit .

cp $root_dir/*_cpp_circuit*.sh .
