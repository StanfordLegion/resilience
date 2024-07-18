#!/bin/sh
#SBATCH --constraint=gpu
#SBATCH --dependency=singleton
#SBATCH --job-name=replay_test
#SBATCH --time=01:10:00
#SBATCH --mail-type=ALL

root_dir="$PWD"

export LD_LIBRARY_PATH="$PWD"

ulimit -S -c 0 # disable core dumps

experiment_name="$(basename "$root_dir")"

nodes=$SLURM_JOB_NUM_NODES
power=$(echo "l($nodes)/l(2)" | bc -l | xargs printf '%.0f\n')

# 3000 iterations runs about 60 seconds, so we're going to do
# 20 * 3000 = 30000 iterations to run about 20 minutes

num_checkpoints=20

if [[ ! -d checkpoint ]]; then mkdir checkpoint; fi
pushd checkpoint

for i in $power; do
  n=$(( 2 ** i))
  nx=$(( 2 ** ((i+1)/2) ))
  ny=$(( 2 ** (i/2) ))

  freq=3000
  slug="${n}x1_f${freq}_orig"
  echo "Running $slug"
  checkpoint_dir="$SCRATCH/$experiment_name/$slug"
  mkdir -p "$checkpoint_dir"

  srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none "$root_dir/stencil.checkpoint" -nx $(( nx * 15000 )) -ny $(( ny * 15000 )) -ntx $(( nx )) -nty $(( ny )) -tsteps $(( num_checkpoints * 3000 )) -tprune 100 -hl:sched 1024 -ll:gpu 1 -ll:io 1 -ll:util 2 -ll:bgwork 4 -ll:csize 10000 -ll:fsize 10000 -ll:zsize 30000 -ll:rsize 0 -ll:gsize 0 -lg:eager_alloc_percentage 10 -lg:no_tracing -level 3 -logfile log_"$slug"_%.log -checkpoint:prefix "$checkpoint_dir" -checkpoint:auto_steps $freq | tee out_"$slug".out

  for replay in $(seq 0 $(( num_checkpoints - 1 )) ); do
    for rep in 0; do
      slug="${n}x1_f${freq}_replay${replay}_r${rep}"
      echo "Running $slug"
      srun -n $n -N $n --ntasks-per-node 1 --cpu_bind none "$root_dir/stencil.checkpoint" -nx $(( nx * 15000 )) -ny $(( ny * 15000 )) -ntx $(( nx )) -nty $(( ny )) -tsteps $(( num_checkpoints * 3000 )) -tprune 100 -hl:sched 1024 -ll:gpu 1 -ll:io 1 -ll:util 2 -ll:bgwork 4 -ll:csize 10000 -ll:fsize 10000 -ll:zsize 30000 -ll:rsize 0 -ll:gsize 0 -lg:eager_alloc_percentage 10 -lg:no_tracing -level 3 -logfile log_"$slug"_%.log -checkpoint:prefix "$checkpoint_dir" -checkpoint:replay $replay -checkpoint:auto_steps $freq -checkpoint:measure_replay_time_and_exit | tee out_"$slug".out
    done
  done

  # # Clean up checkpoints, otherwise we use too much space
  # rm -rf "$checkpoint_dir"
done

popd
