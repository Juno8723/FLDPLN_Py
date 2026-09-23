#!/bin/bash
# Submit the FLDPLN segmentation stage (pyfldpln-segmentation) to SLURM.
# Single self-contained file: resource request + environment + command.
#
# Usage (login node, from anywhere):
#   bash submit_seg.sh                          # default config below
#   bash submit_seg.sh configs/other.yaml       # a different config
#
# IMPORTANT: --cpus-per-task below must equal `run.n_workers` in the config
# (n_workers: null would use the node's total core count, not this allocation).
# Segments already in segment_files/ are skipped, so re-running this script
# resumes a run that hit the time limit.

set -e

cd "$(dirname "$(readlink -f "$0")")"
mkdir -p logs   # SLURM opens the log file before the job starts

CONFIG="${1:-configs/config.03160112.yaml}"

sbatch --export=ALL,CONFIG="$CONFIG" << 'JOB'
#!/bin/bash
#SBATCH --job-name=fldpln_seg
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PyFLDPLN

# Forked segment workers must not each spawn BLAS/OpenMP threads on top.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_CACHE_DIR="$HOME/.numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

cd "$SLURM_SUBMIT_DIR"

echo "host=$(hostname) job=$SLURM_JOB_ID cpus-per-task=$SLURM_CPUS_PER_TASK"
echo "config=$CONFIG"
grep -n "n_workers" "$CONFIG" || true
echo "start=$(date)"

pyfldpln-segmentation "$CONFIG"

echo "end=$(date)"
JOB

echo
echo "check status:  squeue -u \$USER"
echo "follow log:    tail -f logs/fldpln_seg_<jobid>.out"
