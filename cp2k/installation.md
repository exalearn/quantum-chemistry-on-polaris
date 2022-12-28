# Installing on Polaris

We compile CP2K using the GNU compiler suite and CUDA 11.4 on Polaris. 

The install script I use builds most libraries from scratch and links to 

```bash
# Set up the modules to get GNU compilers and CUDA 11.4
#  We choose CUDA 11.4 to be compatible with drivers on the compute nodes
#  We also load the cray implementations of FFTW and LibSci
module reset
module swap PrgEnv-nvhpc PrgEnv-gnu
module load cray-fftw
module load cudatoolkit-standalone/11.4.4
module load cray-libsci
module list

# Define CUDA_PATH, needed by the ELPA build script
export CUDA_PATH=/soft/compilers/cudatoolkit/cuda-11.4.4

# Debug the environmental variables
echo $NVIDIA_PATH
echo $LD_LIBRARY_PATH

# Make the dependencies
cd tools/toolchain
./install_cp2k_toolchain.sh --gpu-ver=A100 --enable-cuda --mpi-mode=mpich | tee install.log
cp install/arch/* ../../arch/
cd ../../

# Make the code
source ./tools/toolchain/install/setup
make -j 16 ARCH=local VERSION="ssmp psmp"
make -j 16 ARCH=local_cuda VERSION="ssmp psmp"
```

## Running CP2K

Running CP2K requires re-establishing the environment,
running the code with the [GPU affinity](https://docs.alcf.anl.gov/polaris/queueing-and-running-jobs/example-job-scripts/#setting-mpi-gpu-affinity)
script provided by ALCF,
and enforcing CPU affinity for the OpenMP portions of the code within the `mpiexec` invocation.


```bash
# Establish the total node count
NNODES=`wc -l < $PBS_NODEFILE`

# Set up the environment
module reset
module swap PrgEnv-nvhpc PrgEnv-gnu
module load conda
module load cudatoolkit-standalone/11.4.4
module load cray-libsci cray-fftw

# Launch CP2k
mpiexec -n $((NNODES * 4)) --ppn 4 --cpu-bind depth --depth 8 -env OMP_NUM_THREADS=8 \
    /lus/grand/projects/CSC249ADCD08/quantum-chemistry-on-polaris/cp2k/mt-polaris-serial/set_affinity_gpu_polaris.sh
    /lus/grand/projects/CSC249ADCD08/cp2k/cp2k-git/exe/local_cuda/cp2k_shell.psmp
```

You can run the serial version by calling the `ssmp` version of CP2K and pinning it to a specific GPU

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 /lus/grand/projects/CSC249ADCD08/cp2k/cp2k-git/exe/local_cuda/cp2k_shell.psmp
```

    Parsl will automatically set the CUDA and OMP environment variables [if patched](https://github.com/Parsl/parsl/pull/2529)
