#!/bin/bash

set -e

for f in src/*.h src/*/*.h src/*/*.cc src/*/*.inl tests/*.cc; do
    # Skip files that are imported from external sources.
    if [[ ! $f = src/resilience/resilience_c.* && \
	  ! $f = src/resilience/resilience_c_util.* ]]; then
        clang-format -i "$f" &
    fi
done
wait

if [[ $CHECK_FORMAT -eq 1 ]]; then
    set -x
    git status
    git diff
    git diff-index --quiet HEAD
fi
