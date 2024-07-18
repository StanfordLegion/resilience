#!/bin/bash
#SBATCH -A CMB103
#SBATCH --partition=batch
#SBATCH --dependency=singleton
#SBATCH --job-name=stencil_test
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL
#SBATCH -C nvme

root_dir="$PWD"

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD"
export SCRATCH="$MEMBERWORK/cmb103"

export FI_MR_CACHE_MONITOR=memhooks
export FI_CXI_RX_MATCH_MODE=software
export GASNET_OFI_DEVICE_0=cxi2
export GASNET_OFI_DEVICE_1=cxi1
export GASNET_OFI_DEVICE_2=cxi3
export GASNET_OFI_DEVICE_3=cxi0
export GASNET_OFI_DEVICE_TYPE=Node
export GASNET_OFI_NUM_RECEIVE_BUFFS=32M

ulimit -S -c 0 # disable core dumps

slurm_flags=
if [[ $SLURM_JOB_NUM_NODES -eq 1 ]]; then
  slurm_flags="--network=single_node_vni"
fi

experiment_name="$(basename "$root_dir")"

nodes=$SLURM_JOB_NUM_NODES
power=$(echo "l($nodes)/l(2)" | bc -l | xargs printf '%.0f\n')

ranks_per_node=8
rank_power=3

if [[ ! -d checkpoint ]]; then mkdir checkpoint; fi
pushd checkpoint

for i in $power; do
  n=$(( 2 ** i))
  nx=$(( (2 ** ((i+1+rank_power)/2)) ))
  ny=$(( (2 ** ((i+rank_power)/2)) ))
  for freq in 3000 1000 300; do
    for r in 0 1 2 3 4; do
      slug="${n}x${ranks_per_node}_f${freq}_r${r}"
      echo "Running $slug"
      checkpoint_dir="$SCRATCH/$experiment_name/$slug"
      set -x
      mkdir -p "$checkpoint_dir"
      srun -n $(( n * ranks_per_node )) -N $n --ntasks-per-node $ranks_per_node --cpus-per-task $(( 56 / ranks_per_node )) --gpus-per-task $(( 8 / ranks_per_node )) --cpu_bind cores $slurm_flags "$root_dir/stencil.checkpoint" -nx $(( nx * 15000 )) -ny $(( ny * 15000 )) -ntx $(( nx )) -nty $(( ny )) -tsteps 3000 -tprune 100 -hl:sched 1024 -ll:gpu 1 -ll:io 1 -ll:util 2 -ll:bgwork 2 -ll:csize 10000 -ll:fsize 10000 -ll:zsize 30000 -ll:rsize 0 -ll:gsize 0 -lg:eager_alloc_percentage 10 -lg:no_tracing -level 3 -logfile log_"$slug"_%.log -checkpoint:prefix "$checkpoint_dir" -checkpoint:auto_steps $freq | tee out_"$slug".out
      # -dm:memoize -lg:parallel_replay 2
      { set +x; } 2>/dev/null

      # Clean up frequent checkpoints, otherwise we use too much space
      # if (( freq < 3000 )); then
          rm -rf "$checkpoint_dir"
      # fi
    done
  done
done

popd

if [[ ! -d no_checkpoint ]]; then mkdir no_checkpoint; fi
pushd no_checkpoint

for i in $power; do
  n=$(( 2 ** i))
  nx=$(( 2 ** ((i+1+rank_power)/2) ))
  ny=$(( 2 ** ((i+rank_power)/2) ))
  for r in 0 1 2 3 4; do
    freq=0
    slug="${n}x${ranks_per_node}_f${freq}_r${r}"
    echo "Running $slug"
    set -x
    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none "$root_dir/stencil.checkpoint" -nx $(( nx * 15000 )) -ny $(( ny * 15000 )) -ntx $(( nx )) -nty $(( ny )) -tsteps 3000 -tprune 100 -hl:sched 1024 -ll:gpu 1 -ll:io 1 -ll:util 2 -ll:bgwork 2 -ll:csize 10000 -ll:fsize 10000 -ll:zsize 30000 -ll:rsize 0 -ll:gsize 0 -lg:eager_alloc_percentage 10 -lg:no_tracing -level 3 -logfile log_"$slug"_%.log -checkpoint:disable | tee out_"$slug".out
    # -dm:memoize -lg:parallel_replay 2
    { set +x; } 2>/dev/null
  done
done

popd
