#!/bin/bash

set -e

tmp_dir=$(mktemp -d)
cd $tmp_dir

set -x

# Run once normally. Generate all checkpoints.
"$@"

# Make sure every checkpoint is individually replayable.
for checkpoint in *.dat; do
  "$@" -replay -cpt $(echo "$checkpoint" | cut -d. -f2)
done
