#!/bin/bash

set -e
set -x

tmp_dir=$(mktemp -d)
cd $tmp_dir

# Run once normally. Generate all checkpoints.
"$@"

# Make sure every checkpoint is individually replayable.
check=0
if compgen -G '*.dat' > /dev/null; then
    for checkpoint in *.dat; do
        if [[ $checkpoint = *".lr."*".dat" ]]; then
            echo "Skipping region checkpoint $checkpoint"
        else
            "$@" -replay -cpt $(echo "$checkpoint" | cut -d. -f2)
            check=1
        fi
    done
fi
if [[ $check -eq 0 ]]; then
    echo "No checkpoints in normal run"
    exit 1
fi

rm -f *.dat

# Run with abort (if supported).
"$@" -abort || true

# Make sure every checkpoint is individually replayable.
if compgen -G '*.dat' > /dev/null; then
    for checkpoint in *.dat; do
        if [[ $checkpoint = *".lr."*".dat" ]]; then
            echo "Skipping region checkpoint $checkpoint"
        else
            "$@" -replay -cpt $(echo "$checkpoint" | cut -d. -f2)
            check=1
        fi
    done
fi
