#!/bin/bash
# Submit the FLDPLN gauge flood-map stage (pyfldpln-floodmap) to SLURM.
# Run this only after pyfldpln-library-reorganization (submit_lib.sh) has
# completed -- it reads library/ written by that stage. Safe to re-run
# repeatedly with a fresh gauges CSV; unlike the other two stages this one
# is meant to be called on a schedule, not once per AOI.
#
# Usage (login node, from anywhere):
#   bash submit_floodmap.sh
#   bash submit_floodmap.sh <config.yaml> <gauges.csv> <snap_dist> [out.tif]
#
# Defaults below match this AOI's setup. `gauges.csv` needs x, y (library's
# own CRS, meters) and stage_elevation (same vertical datum/units as
# fsp_info.csv's FilledElev -- feet) columns; unlike the config's own
# paths.*, this path is NOT resolved against paths.home_dir, so pass it
# absolute or relative to wherever you run this script from.
#
# --snap-dist: max gauge-to-flood-source-point distance, in the library's
# CRS units (meters here). 350 is carried over from the project's Amite
# example run, NOT validated for this AOI's gauge spacing -- check how many
# gauges actually get used in the output log and widen/narrow if that looks
# wrong (e.g. most/all of the 8 gauges in study_03160112_gauges.csv should
# snap to something on the Black Warrior mainstem).
#
# This stage does no multiprocessing (like submit_lib.sh) and is lighter
# than library-reorganization was (that took ~31s / ~1.1GB) -- resources
# below are a conservative first guess; check sacct and shrink further.

set -e

cd "$(dirname "$(readlink -f "$0")")"
mkdir -p logs   # SLURM opens the log file before the job starts

CONFIG="${1:-configs/config.03160112.yaml}"
GAUGES="${2:-bil_03160112/Gage/study_03160112_gauges.csv}"
SNAP_DIST="${3:-350}"
OUT="${4:-}"

sbatch --export=ALL,CONFIG="$CONFIG",GAUGES="$GAUGES",SNAP_DIST="$SNAP_DIST",OUT="$OUT" << 'JOB'
#!/bin/bash
#SBATCH --job-name=fldpln_floodmap
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
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
echo "config=$CONFIG gauges=$GAUGES snap_dist=$SNAP_DIST out=${OUT:-<default>}"
echo "start=$(date)"

if [ -n "$OUT" ]; then
    pyfldpln-floodmap "$CONFIG" "$GAUGES" --snap-dist "$SNAP_DIST" --out "$OUT"
else
    pyfldpln-floodmap "$CONFIG" "$GAUGES" --snap-dist "$SNAP_DIST"
fi

echo "end=$(date)"
JOB

echo
echo "check status:  squeue -u \$USER"
echo "follow log:    tail -f logs/fldpln_floodmap_<jobid>.out"
