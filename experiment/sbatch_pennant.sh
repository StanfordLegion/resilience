#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL

root_dir="$PWD"

export LD_LIBRARY_PATH="$PWD"

ulimit -S -c 0 # disable core dumps

experiment_name="$(basename "$root_dir")"

if [[ ! -d checkpoint ]]; then mkdir checkpoint; fi
pushd checkpoint

for n in $SLURM_JOB_NUM_NODES; do
  for r in 0 1 2 3 4; do
    echo "Running $n""x1_r$r"
    checkpoint_dir="$SCRATCH/$experiment_name/${n}x1_r${r}"
    mkdir -p "$checkpoint_dir"
    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none "$root_dir/pennant.checkpoint" "$root_dir"/pennant.tests/leblanc_long"$(( n * 4 ))"x30/leblanc.pnt -npieces "$n" -numpcx 1 -numpcy "$n" -seq_init 0 -par_init 1 -prune 30 -hl:sched 1024 -ll:gpu 1 -ll:util 2 -ll:bgwork 2 -ll:csize 15000 -ll:fsize 15000 -ll:zsize 2048 -ll:rsize 512 -ll:gsize 0 -lg:eager_alloc_percentage 10 -level 3 -dm:memoize -lg:parallel_replay 2 -checkpoint:prefix "$checkpoint_dir" | tee out_"$n"x1_r"$r".log
  done
done

popd

if [[ ! -d no_checkpoint ]]; then mkdir no_checkpoint; fi
pushd no_checkpoint

for n in $SLURM_JOB_NUM_NODES; do
  for r in 0 1 2 3 4; do
    echo "Running $n""x1_r$r"
    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none "$root_dir/pennant.checkpoint" "$root_dir"/pennant.tests/leblanc_long"$(( n * 4 ))"x30/leblanc.pnt -npieces "$n" -numpcx 1 -numpcy "$n" -seq_init 0 -par_init 1 -prune 30 -hl:sched 1024 -ll:gpu 1 -ll:util 2 -ll:bgwork 2 -ll:csize 15000 -ll:fsize 15000 -ll:zsize 2048 -ll:rsize 512 -ll:gsize 0 -lg:eager_alloc_percentage 10 -level 3 -dm:memoize -lg:parallel_replay 2 -checkpoint:disable | tee out_"$n"x1_r"$r".log
  done
done

popd
