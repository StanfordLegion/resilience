export MACHINE=frontier
export THREADS=20

module swap $LMOD_FAMILY_PRGENV PrgEnv-gnu
module load rocm/6.0.0
module unload darshan-runtime

export USE_HIP=1
export CONDUIT=ofi-slingshot11

export REGENT_GPU_FLAGS="-fgpu hip -fgpu-arch gfx90a"

export THRUST_PATH="$PWD/Thrust"
