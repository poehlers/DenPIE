#!/bin/bash
#SBATCH --job-name=spectra_SBI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=8:00:00
#SBATCH --output=/home/poehlers/cosmo_thesis/den_pie/output/logs/%x-%j-%N_slurm.out
#SBATCH --error=/home/poehlers/cosmo_thesis/den_pie/output/logs/R-%x.%j.err

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load OpenMPI/5.0.7-GCC-14.2.0
module load FFTW/3.3.10-GCC-14.2.0

source /home/poehlers/21cmvenv/bin/activate

cd /home/poehlers/cosmo_thesis/den_pie

python -m den_pie spectra-train params/spectra_train_fli_bias_SN.yaml --verbose
python -m den_pie spectra-plot  params/spectra_train_fli_bias_SN.yaml --verbose
