export MACHINE=perlmutter
export THREADS=20

module load PrgEnv-gnu
module load cpe-cuda
module load cudatoolkit

export USE_CUDA=1
export CONDUIT=ofi-slingshot11

export REGENT_GPU_FLAGS=(-fcuda 1 -fcuda-offline 1 -fcuda-arch ampere -fcuda-generate-cubin 1)
