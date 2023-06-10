#!/bin/bash
## SLURM scripts have a specific format. 

### Section1: SBATCH directives to specify job configuration
#SBATCH --job-name=vq2d_val_full_dino_new_hyp
#SBATCH --output=/scratch/shared/beegfs/prannay/ego4d_data/logs/slurm_eval_new_new/eval_vq2d-%j.out
#####ASBATCH --error=/scratch/shared/beegfs/prannay/ego4d_data/logs/slurm_eval_new_new/eval_vq2d-%j.err
#SBATCH --partition="low-prio-gpu"
#SBATCH --array=1-199
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --constraint="a40|a6000|rtx6k"
#SBATCH --comment="vq2d trace baseline evaluation"
#SBATCH --exclude="gnodek3"
#SBATCH --mail-type="ALL"
#SBATCH --mail-user="prannay.kaul@gmail.com"

# echo `localhost`

source ~/.bashrc
source ~/.bash_profile

### Section 2: Setting environment variables for the job
source ~/miniconda3/etc/profile.d/conda.sh

### Section 3:
# conda deactivate
# source activate ego4d_vq2d
# conda deactivate
# conda deactivate

conda activate vq2d

VQ2D_ROOT=/users/prannay/vq2d/vq2d_cvpr
# /private/home/frostxu/VQ2D_CVPR22/checkpoint/train_log/slurm_8gpus_4nodes_v11_set_bijit_t128_rmgt0.5_frame_llr
TRAIN_ROOT=/scratch/shared/beegfs/prannay/ego4d_data/ckpt
EVAL_ROOT=$TRAIN_ROOT/full_run/dino_rerank_new_hyp
CLIPS_ROOT=/scratch/shared/beegfs/prannay/ego4d_data/vq2d_clips_val

mkdir -p $EVAL_ROOT

VQ2D_SPLITS_ROOT=$VQ2D_ROOT/data/
PYTRACKING_ROOT=$VQ2D_ROOT/dependencies/pytracking

N_PART=200.0
ITER='0064999'

export PYTHONPATH="$PYTHONPATH:$VQ2D_ROOT"
export PYTHONPATH="$PYTHONPATH:$PYTRACKING_ROOT"
export MAX_JOBS=4

cd $VQ2D_ROOT
which python

sleep $((RANDOM%30+1))

# SLURM_ARRAY_TASK_ID is from 1 to 100
# SLURM_ARRAY_TASK_ID=1.0
python evaluate_vq2d_tracker_pk2_trace_v2_all.py \
  data.data_root="$CLIPS_ROOT" \
  data.split="val" \
  +data.part=$SLURM_ARRAY_TASK_ID \
  +data.n_part=$N_PART \
  data.annot_root="$VQ2D_SPLITS_ROOT" \
  data.num_processes=1 \
  num_gpus=1 \
  num_machines=1 \
  machine_rank=0 \
  data.debug_mode=False \
  data.num_workers=0 \
  +model.num_token=128 \
  data.rcnn_batch_size=16 \
  +signals.height=0.4 \
  signals.width=3 \
  signals.prominence=0.2 \
  signals.distance=3 \
  model.config_path="$TRAIN_ROOT/config.yaml" \
  model.checkpoint_path="$TRAIN_ROOT/model_${ITER}.pth" \
  logging.save_dir="$EVAL_ROOT" \
  logging.stats_save_path="$EVAL_ROOT/vq_stats_val_$SLURM_ARRAY_TASK_ID.json.gz" \
  +data.prior_pred_path="/scratch/shared/beegfs/prannay/ego4d_data/ckpt/results/traces_v2_all/" \
  +dino_rerank=True

# echo $EVAL_ROOT/model_${ITER}
#   # signals.distance=5 signals.width=3
