#!/bin/bash

set -e

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

mkdir "$1"
cd "$1"

cp $root_dir/../legion/install/lib*/*.so* .
cp $root_dir/../build/examples/pennant/pennant .

cp $root_dir/$MACHINE/*_cpp_pennant*.sh .
cp -r $root_dir/pennant.tests .
