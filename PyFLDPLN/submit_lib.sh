#!/bin/bash
# Submit the FLDPLN library-reorganization stage (pyfldpln-library-reorganization)
# to SLURM. Run this only after pyfldpln-segmentation (submit_seg.sh) has
# completed -- it reads segment_files/ written by that stage.
#
# Usage (login node, from anywhere):
#   bash submit_lib.sh                          # default config below
#   bash submit_lib.sh configs/other.yaml       # a different config
#
# Unlike segmentation, this stage does NOT use run.n_workers -- it's a single
# in-process pass (FSP table -> stream order -> per-segment relations ->
# tiling), no multiprocessing/threading. --cpus-per-task is left at 4 only
# as headroom for numpy/pandas' own internal vectorized ops, not for worker
# parallelism -- there's no n_workers value to match here.
#
# --mem/--time below are untested placeholders (this stage hasn't been timed
# on HPC or at this AOI's scale yet) -- check sacct after this first run and
# tighten them, the same way submit_seg.sh's --mem=128G turned out to be
# ~7x more than the ~17GB actually used.

set -e

cd "$(dirname "$(readlink -f "$0")")"
mkdir -p logs   # SLURM opens the log file before the job starts

CONFIG="${1:-configs/config.03160112.yaml}"

sbatch --export=ALL,CONFIG="$CONFIG" << 'JOB'
#!/bin/bash
#SBATCH --job-name=fldpln_lib
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PyFLDPLN

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"

echo "host=$(hostname) job=$SLURM_JOB_ID cpus-per-task=$SLURM_CPUS_PER_TASK"
echo "config=$CONFIG"
echo "start=$(date)"

pyfldpln-library-reorganization "$CONFIG"

echo "end=$(date)"
JOB

echo
echo "check status:  squeue -u \$USER"
echo "follow log:    tail -f logs/fldpln_lib_<jobid>.out"
