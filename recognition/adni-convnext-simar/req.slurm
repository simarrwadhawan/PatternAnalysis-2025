#!/bin/bash
#SBATCH --job-name=SW
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=s4977354@uq.edu.au
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH -o predict.out
#SBATCH -e predict.err
#SBATCH --partition=a100
#SBATCH --time=0:30:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate comp3710

echo "Starting model evaluation and visualization generation..."
echo "=========================================================="
echo "Using improved model with class weights (98.82% val accuracy)"
echo ""

python predict.py \
    --checkpoint runs/adni_convnext_balanced/best.pt \
    --data-root /home/groups/comp3710/ADNI/AD_NC \
    --model convnext_tiny \
    --batch-size 32 \
    --img-size 224 \
    --save-dir ./images

echo ""
echo "=========================================================="
echo "Evaluation complete!"
echo "Check ./images/ folder for visualizations"
echo "Expected: ~95-98% test accuracy (up from 73.81%)"
