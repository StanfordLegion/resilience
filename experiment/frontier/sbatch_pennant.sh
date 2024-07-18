#!/bin/bash
#SBATCH -A CMB103
#SBATCH --partition=batch
#SBATCH --dependency=singleton
#SBATCH --job-name=circuit_test
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL
#SBATCH -C nvme

root_dir="$PWD"

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD"
export SCRATCH="$MEMBERWORK/cmb103"

ulimit -S -c 0 # disable core dumps

slurm_flags=
if [[ $SLURM_JOB_NUM_NODES -eq 1 ]]; then
  slurm_flags="--network=single_node_vni"
fi

experiment_name="$(basename "$root_dir")"

ranks_per_node=8

if [[ ! -d checkpoint ]]; then mkdir checkpoint; fi
pushd checkpoint

for n in $SLURM_JOB_NUM_NODES; do
  for freq in 1000 300 100; do
    for r in 0 1 2 3 4; do
      slug="${n}x${ranks_per_node}_f${freq}_r${r}"
      echo "Running $slug"
      checkpoint_dir="$SCRATCH/$experiment_name/$slug"
      set -x
      mkdir -p "$checkpoint_dir"
      srun -n $(( n * ranks_per_node )) -N $n --ntasks-per-node $ranks_per_node --cpus-per-task $(( 56 / ranks_per_node )) --gpus-per-task $(( 8 / ranks_per_node )) --cpu_bind cores $slurm_flags "$root_dir/pennant.checkpoint" "$root_dir"/pennant.tests/leblanc_long"$(( n * ranks_per_node * 4 ))"x1000/leblanc.pnt -npieces "$(( n * ranks_per_node ))" -numpcx 1 -numpcy "$(( n * ranks_per_node ))" -seq_init 0 -par_init 1 -prune 30 -hl:sched 1024 -ll:gpu 1 -ll:io 1 -ll:util 2 -ll:bgwork 4 -ll:csize 12000 -ll:fsize 12000 -ll:zsize 36000 -ll:rsize 0 -ll:gsize 0 -lg:eager_alloc_percentage 10 -lg:no_tracing -level 3 -logfile log_"$slug"_%.log -checkpoint:prefix "$checkpoint_dir" -checkpoint:auto_steps $freq | tee out_"$slug".out
      # -dm:memoize -lg:parallel_replay 2
      { set +x; } 2>/dev/null

      # Clean up frequent checkpoints, otherwise we use too much space
      # if (( freq < 1000 )); then
          rm -rf "$checkpoint_dir"
      # fi
    done
  done
done

popd

if [[ ! -d no_checkpoint ]]; then mkdir no_checkpoint; fi
pushd no_checkpoint

for n in $SLURM_JOB_NUM_NODES; do
  for r in 0 1 2 3 4; do
    freq=0
    slug="${n}x${ranks_per_node}_f${freq}_r${r}"
    echo "Running $slug"
    set -x
    srun -n $(( n * ranks_per_node )) -N $n --ntasks-per-node $ranks_per_node --cpus-per-task $(( 56 / ranks_per_node )) --gpus-per-task $(( 8 / ranks_per_node )) --cpu_bind cores $slurm_flags "$root_dir/pennant.checkpoint" "$root_dir"/pennant.tests/leblanc_long"$(( n * ranks_per_node * 4 ))"x1000/leblanc.pnt -npieces "$(( n * ranks_per_node ))" -numpcx 1 -numpcy "$(( n * ranks_per_node ))" -seq_init 0 -par_init 1 -prune 30 -hl:sched 1024 -ll:gpu 1 -ll:io 1 -ll:util 2 -ll:bgwork 2 -ll:csize 12000 -ll:fsize 12000 -ll:zsize 36000 -ll:rsize 0 -ll:gsize 0 -lg:eager_alloc_percentage 10 -lg:no_tracing -level 3 -logfile log_"$slug"_%.log -checkpoint:disable | tee out_"$slug".out
    # -dm:memoize -lg:parallel_replay 2
    { set +x; } 2>/dev/null
  done
done

popd
