#!/bin/sh
#SBATCH --constraint=gpu
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL

root_dir="$PWD"

export LD_LIBRARY_PATH="$PWD"

ulimit -S -c 0 # disable core dumps

experiment_name="$(basename "$root_dir")"

# 300 iterations runs about 10 seconds, so we're going to do
# 60 * 300 = 18000 iterations to run about 10 minutes

num_checkpoints=60

if [[ ! -d checkpoint ]]; then mkdir checkpoint; fi
pushd checkpoint

for n in $SLURM_JOB_NUM_NODES; do
  freq=300
  slug="${n}x1_f${freq}_orig"
  echo "Running $slug"
  checkpoint_dir="$SCRATCH/$experiment_name/$slug"
  mkdir -p "$checkpoint_dir"

  srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none "$root_dir/circuit.checkpoint" -npp 5000 -wpp 20000 -l $(( num_checkpoints * 300 )) -p $(( $n * 10 )) -pps 10 -prune 30 -hl:sched 1024 -ll:gpu 1 -ll:io 1 -ll:util 2 -ll:bgwork 4 -ll:csize 15000 -ll:fsize 15000 -ll:zsize 15000 -ll:rsize 0 -ll:gsize 0 -lg:eager_alloc_percentage 10 -lg:no_tracing -level 3 -logfile log_"$slug"_%.log -checkpoint:prefix "$checkpoint_dir" -checkpoint:auto_steps $freq | tee out_"$slug".out

  for r in $(seq 0 $(( num_checkpoints - 1 )) ); do
    slug="${n}x1_f${freq}_replay${r}"
    echo "Running $slug"
    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none "$root_dir/circuit.checkpoint" -npp 5000 -wpp 20000 -l $(( num_checkpoints * 300 )) -p $(( $n * 10 )) -pps 10 -prune 30 -hl:sched 1024 -ll:gpu 1 -ll:io 1 -ll:util 2 -ll:bgwork 4 -ll:csize 15000 -ll:fsize 15000 -ll:zsize 15000 -ll:rsize 0 -ll:gsize 0 -lg:eager_alloc_percentage 10 -lg:no_tracing -level 3 -logfile log_"$slug"_%.log -checkpoint:prefix "$checkpoint_dir" -checkpoint:replay $r -checkpoint:auto_steps $freq -checkpoint:measure_replay_time_and_exit | tee out_"$slug".out
  done

  # Clean up checkpoints, otherwise we use too much space
  rm -rf "$checkpoint_dir"
done

popd
