#!/bin/bash
## slurm-eagle
## number of nodes
#SBATCH --nodes 1

## task per node
#SBATCH --tasks-per-node=28

## wall time in format MINUTES:SECONDS
#SBATCH --time=0-0:59:00

## grant
#SBATCH --account=vecma2020

## stdout file
#SBATCH --output=/home/plgrid/plgwedeling/FabSim3/results/PC_CI_HQ_SD_suppress_campaign3_1_repeat_eagle_vecma_28/RUNS/Run_160/JobID-%j.output

## stderr file
#SBATCH --error=/home/plgrid/plgwedeling/FabSim3/results/PC_CI_HQ_SD_suppress_campaign3_1_repeat_eagle_vecma_28/RUNS/Run_160/JobID-%j.error

## Memory limit per compute node for the job.
## Do not use with mem-per-cpu flag.
#SBATCH --mem=20GB

#SBATCH --partition=fast



cd /home/plgrid/plgwedeling/FabSim3/results/PC_CI_HQ_SD_suppress_campaign3_1_repeat_eagle_vecma_28/RUNS/Run_160
module load python/3.7.3 && module load r/3.6.1-gcc620

/usr/bin/env > env.log

export OMP_NUM_THREADS=28

start_time="$(date -u +%s.%N)"

python3 run_sample.py --outputdir output_dir --threads 28 United_Kingdom 2>&1 | tee output_log.txt

end_time="$(date -u +%s.%N)"
elapsed="$(bc <<<"$end_time-$start_time")"
echo "Total Executing Time = $elapsed seconds" | tee -a "elapsed_time.txt"

# convert all xls file to csv
# Rscript Rscripts/Convert_xls_to_csv.R output_dir

# Run visualisation with R
Rscript Rscripts/PlotRuns.R output_dir
Rscript Rscripts/CompareRuns.R output_dir

