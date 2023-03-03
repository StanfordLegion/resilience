#!/bin/bash

set -e

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

mkdir "$1"
cd "$1"

SAVEOBJ=1 STANDALONE=1 OBJNAME=./pennant.checkpoint $root_dir/../legion/language/regent.py $root_dir/../legion/language/examples/pennant.rg -fflow 0 -fopenmp 0 -fcuda 1 -fcuda-offline 1 -fcuda-arch pascal -fcheckpoint 1

cp $root_dir/*_pennant*.sh .
