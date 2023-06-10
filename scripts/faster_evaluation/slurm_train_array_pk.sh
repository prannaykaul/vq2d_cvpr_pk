#!/bin/bash
## SLURM scripts have a specific format. 

### Section1: SBATCH directives to specify job configuration
#SBATCH --job-name=vq2d_train_trace
#SBATCH --output=/scratch/shared/beegfs/prannay/ego4d_data/logs/slurm_eval/train_vq2d-%j.out
#####ASBATCH --error=/scratch/shared/beegfs/prannay/ego4d_data/logs/slurm_eval/eval_vq2d-%j.err
#SBATCH --partition="low-prio-gpu"
#SBATCH --array=0-99
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --constraint "p40|rtx6k|a40|a4500|rtx8k|v100s"
#SBATCH --mem=56GB
#SBATCH --time=12:00:00
#SBATCH --comment="vq2d trace baseline evaluation"

# echo `localhost`

### Section 2: Setting environment variables for the job
source ~/miniconda3/etc/profile.d/conda.sh

### Section 3:
# conda deactivate
# source activate ego4d_vq2d
# conda deactivate
# conda deactivate
conda activate vq2d
. ~/.bashrc
. ~/.bash_profile

VQ2D_ROOT=$PWD
# /private/home/frostxu/VQ2D_CVPR22/checkpoint/train_log/slurm_8gpus_4nodes_v11_set_bijit_t128_rmgt0.5_frame_llr
TRAIN_ROOT=/scratch/shared/beegfs/prannay/ego4d_data/ckpt
EVAL_ROOT=$TRAIN_ROOT/results/traces_train
CLIPS_ROOT=/scratch/shared/beegfs/prannay/ego4d_data/vq2d_clips_train

mkdir -p $EVAL_ROOT

VQ2D_SPLITS_ROOT=$VQ2D_ROOT/data/
PYTRACKING_ROOT=$VQ2D_ROOT/dependencies/pytracking

N_PART=100.0
ITER='0064999'

export PYTHONPATH="$PYTHONPATH:$VQ2D_ROOT"
export PYTHONPATH="$PYTHONPATH:$PYTRACKING_ROOT"

cd $VQ2D_ROOT
which python

sleep $((RANDOM%30+1))
# SLURM_ARRAY_TASK_ID is from 1 to 100
# SLURM_ARRAY_TASK_ID=1.0
python evaluate_vq2d_no_tracker_pk2_trace.py \
  data.data_root="$CLIPS_ROOT" \
  data.split="train" \
  +data.part=$SLURM_ARRAY_TASK_ID \
  +data.n_part=$N_PART \
  data.annot_root="$VQ2D_SPLITS_ROOT" \
  data.num_processes=1 \
  data.debug_mode=False \
  data.num_workers=0 \
  +model.num_token=128 \
  data.rcnn_batch_size=16 \
  +signals.height=0.1 \
  model.config_path="$TRAIN_ROOT/config.yaml" \
  model.checkpoint_path="$TRAIN_ROOT/model_${ITER}.pth" \
  logging.save_dir="$EVAL_ROOT" \
  logging.stats_save_path="$EVAL_ROOT/vq_stats_train_$SLURM_ARRAY_TASK_ID.json.gz"

# echo $EVAL_ROOT/model_${ITER}
#   # signals.distance=5 signals.width=3
