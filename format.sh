#!/bin/bash

set -e

for f in src/*.cc src/*.h tests/*.cc; do
    clang-format -i "$f"
done
