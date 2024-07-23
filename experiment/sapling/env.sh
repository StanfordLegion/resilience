export MACHINE=sapling
export THREADS=20

module load cuda

export USE_CUDA=1
export CONDUIT=ibv

export REGENT_GPU_FLAGS="-fgpu cuda -fgpu-arch pascal"
