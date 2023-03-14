#!/bin/sh
#SBATCH --constraint=gpu
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL

root_dir="$PWD"

export LD_LIBRARY_PATH="$PWD"

ulimit -S -c 0 # disable core dumps

experiment_name="$(basename "$root_dir")"

if [[ ! -d checkpoint ]]; then mkdir checkpoint; fi
pushd checkpoint

for n in $SLURM_JOB_NUM_NODES; do
  for freq in 300 100 30 10; do
    for r in 0 1 2 3 4; do
      slug="${n}x1_f${freq}_r${r}"
      echo "Running $slug"
      checkpoint_dir="$SCRATCH/$experiment_name/$slug"
      mkdir -p "$checkpoint_dir"
      srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none "$root_dir/circuit" -npp 5000 -wpp 20000 -pct 98 -l 350 -p $n -prune 30 -hl:sched 1024 -ll:gpu 1 -ll:io 1 -ll:util 2 -ll:bgwork 4 -ll:csize 15000 -ll:fsize 15000 -ll:zsize 2048 -ll:rsize 512 -ll:gsize 0 -lg:eager_alloc_percentage 10 -lg:no_tracing -level 3 -logfile log_"$slug"_%.log -checkpoint:prefix "$checkpoint_dir" -checkpoint:auto_steps $freq | tee out_"$slug".out
      # -dm:memoize -lg:parallel_replay 2

      # Clean up frequent checkpoints, otherwise we use too much space
      if (( freq < 300 )); then
          rm -rf "$checkpoint_dir"
      fi
    done
  done
done

popd

if [[ ! -d no_checkpoint ]]; then mkdir no_checkpoint; fi
pushd no_checkpoint

for n in $SLURM_JOB_NUM_NODES; do
  for r in 0 1 2 3 4; do
    freq=0
    slug="${n}x1_f${freq}_r${r}"
    echo "Running $slug"
    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none "$root_dir/circuit" -npp 5000 -wpp 20000 -pct 98 -l 350 -p $n -prune 30 -hl:sched 1024 -ll:gpu 1 -ll:io 1 -ll:util 2 -ll:bgwork 4 -ll:csize 15000 -ll:fsize 15000 -ll:zsize 2048 -ll:rsize 512 -ll:gsize 0 -lg:eager_alloc_percentage 10 -lg:no_tracing -level 3 -logfile log_"$slug"_%.log -checkpoint:disable | tee out_"$slug".out
    # -dm:memoize -lg:parallel_replay 2
  done
done

popd
