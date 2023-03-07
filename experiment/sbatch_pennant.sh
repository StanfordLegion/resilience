#!/bin/bash
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
  for freq in 100 30 10 3 1; do
    for r in 0 1 2 3 4; do
      slug="${n}x1_f${freq}_r${r}"
      echo "Running $slug"
      checkpoint_dir="$SCRATCH/$experiment_name/$slug"
      mkdir -p "$checkpoint_dir"
      srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none "$root_dir/pennant.checkpoint" "$root_dir"/pennant.tests/leblanc_long"$(( n * 4 ))"x30/leblanc.pnt -npieces "$n" -numpcx 1 -numpcy "$n" -seq_init 0 -par_init 1 -prune 30 -hl:sched 1024 -ll:gpu 1 -ll:util 2 -ll:bgwork 2 -ll:csize 15000 -ll:fsize 15000 -ll:zsize 2048 -ll:rsize 512 -ll:gsize 0 -lg:eager_alloc_percentage 10 -level 3 -lg:no_tracing -checkpoint:prefix "$checkpoint_dir" -checkpoint:auto_steps $freq -checkpoint:skip_leak_check | tee out_"$slug".log
      # -dm:memoize -lg:parallel_replay 2
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
    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none "$root_dir/pennant.checkpoint" "$root_dir"/pennant.tests/leblanc_long"$(( n * 4 ))"x30/leblanc.pnt -npieces "$n" -numpcx 1 -numpcy "$n" -seq_init 0 -par_init 1 -prune 30 -hl:sched 1024 -ll:gpu 1 -ll:util 2 -ll:bgwork 2 -ll:csize 15000 -ll:fsize 15000 -ll:zsize 2048 -ll:rsize 512 -ll:gsize 0 -lg:eager_alloc_percentage 10 -level 3 -lg:no_tracing -checkpoint:disable | tee out_"$slug".log
    # -dm:memoize -lg:parallel_replay 2
  done
done

popd
