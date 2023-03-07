#!/bin/sh
#SBATCH --constraint=gpu
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL

root_dir="$PWD"

export LD_LIBRARY_PATH="$PWD"

ulimit -S -c 0 # disable core dumps

experiment_name="$(basename "$root_dir")"

nodes=$SLURM_JOB_NUM_NODES
power=$(echo "l($nodes)/l(2)" | bc -l | xargs printf '%.0f\n')

if [[ ! -d checkpoint ]]; then mkdir checkpoint; fi
pushd checkpoint

for i in $power; do
  n=$(( 2 ** i))
  nx=$(( 2 ** ((i+1)/2) ))
  ny=$(( 2 ** (i/2) ))
  for freq in 100 30 10 3 1; do
    for r in 0 1 2 3 4; do
      slug="${n}x1_f${freq}_r${r}"
      echo "Running $slug"
      checkpoint_dir="$SCRATCH/$experiment_name/$slug"
      mkdir -p "$checkpoint_dir"
      srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none "$root_dir/stencil.checkpoint" -nx $(( nx * 20000 )) -ny $(( ny * 20000 )) -ntx $(( nx )) -nty $(( ny )) -tsteps 50 -tprune 30 -hl:sched 1024 -ll:gpu 1 -ll:util 1 -ll:bgwork 2 -ll:csize 15000 -ll:fsize 15000  -ll:rsize 512 -ll:gsize 0 -lg:eager_alloc_percentage 10 -level 3 -lg:no_tracing -checkpoint:prefix "$checkpoint_dir" -checkpoint:auto_steps $freq -checkpoint:skip_leak_check | tee out_"$slug".log
      # -dm:memoize -lg:parallel_replay 2
    done
  done
done

popd

if [[ ! -d no_checkpoint ]]; then mkdir no_checkpoint; fi
pushd no_checkpoint

for i in $power; do
  n=$(( 2 ** i))
  nx=$(( 2 ** ((i+1)/2) ))
  ny=$(( 2 ** (i/2) ))
  for r in 0 1 2 3 4; do
    freq=0
    slug="${n}x1_f${freq}_r${r}"
    echo "Running $slug"
    srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none "$root_dir/stencil.checkpoint" -nx $(( nx * 20000 )) -ny $(( ny * 20000 )) -ntx $(( nx )) -nty $(( ny )) -tsteps 50 -tprune 30 -hl:sched 1024 -ll:gpu 1 -ll:util 1 -ll:bgwork 2 -ll:csize 15000 -ll:fsize 15000  -ll:rsize 512 -ll:gsize 0 -lg:eager_alloc_percentage 10 -level 3 -lg:no_tracing -checkpoint:disable | tee out_"$slug".log
    # -dm:memoize -lg:parallel_replay 2
  done
done

popd
