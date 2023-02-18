#!/bin/bash

set -e
set -x

tmp_dir=$(mktemp -d)
pushd $tmp_dir

function is_root_checkpoint() {
    [[ ! ( $1 = *".lr."*".dat" ) ]]
}

function copy_checkpoint_files() {
    cp "$1/$(basename "$2" .dat)"*.dat .
}

# Run once normally. Generate all checkpoints.
mkdir orig
pushd orig
"$@"
popd # orig

# Make sure every checkpoint is individually replayable.
mkdir orig_replay
pushd orig_replay
check=0
if compgen -G '../orig/*.dat' > /dev/null; then
    for checkpoint in ../orig/*.dat; do
        if is_root_checkpoint "$checkpoint"; then
            rm -f *.dat
            copy_checkpoint_files ../orig "$checkpoint"
            ls -l
            "$@" -replay -cpt $(echo "$(basename "$checkpoint")" | cut -d. -f2)
            ls -l
            check=1
        fi
    done
fi
if [[ $check -eq 0 ]]; then
    echo "No checkpoints in normal run"
    exit 1
fi
popd # orig_replay

# Run with abort (if supported).
mkdir abort
pushd abort
"$@" -abort || true
popd # abort

# Make sure every checkpoint is individually replayable.
mkdir abort_replay
pushd abort_replay
if compgen -G '../abort/*.dat' > /dev/null; then
    for checkpoint in ../abort/*.dat; do
        if is_root_checkpoint "$checkpoint"; then
            rm -f *.dat
            copy_checkpoint_files ../abort "$checkpoint"
            ls -l
            "$@" -replay -cpt $(echo "$(basename "$checkpoint")" | cut -d. -f2)
            ls -l
        fi
    done
fi
popd # abort_replay

popd # $tmp_dir
rm -rf $tmp_dir
