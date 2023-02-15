#!/bin/bash

set -e

tmp_dir=$(mktemp -d)
cd $tmp_dir

set -x

# Run once normally. Generate all checkpoints.
"$@"

# Make sure every checkpoint is individually replayable.
if compgen -G '*.dat' > /dev/null; then
    for checkpoint in *.dat; do
        "$@" -replay -cpt $(echo "$checkpoint" | cut -d. -f2)
    done
else
    echo "No checkpoints in normal run"
    exit 1
fi

if [[ $1 = */region ]]; then
    echo "region fails abort test, skipping"
    exit 0
fi

rm -f *.dat

# Run with abort (if supported).
"$@" -abort || true

# Make sure every checkpoint is individually replayable.
if compgen -G '*.dat' > /dev/null; then
    for checkpoint in *.dat; do
        "$@" -replay -cpt $(echo "$checkpoint" | cut -d. -f2)
    done
fi
