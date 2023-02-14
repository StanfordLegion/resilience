#!/bin/bash

set -e

for f in src/*.cc src/*.h tests/*.cc; do
    clang-format -i "$f"
done

if [[ $CHECK_FORMAT -eq 1 ]]; then
    set -x
    git status
    git diff
    git diff-index --quiet HEAD
fi
