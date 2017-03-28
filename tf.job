#!/bin/bash
#
#SBATCH --job-name=mp
#SBATCH --output=mp.out.%j
#SBATCH --qos=normal
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres gpu:1
#SBATCH --gres-flags=enforce-binding

# XStream

module load tensorflow/1.0.0

mkdir -p $dirname
cd $dirname
cp $WORK/enhance/*.py .
echo $dirname, $i, $lr, $s1, $s2, $s3, $f > config.txt

srun python sres_multi_gpu_train.py \
  --num_gpus=1 \
  --data_dir=/cstor/xsede/users/xs-edwardk/webcam \
  --train_dir=$dirname/logs \
  --upscale_factor=4 \
  --batch_size=16 \
  --initial_learning_rate=$lr \
  --num_filters=$f \
  --first_filter_size=$s1 \
  --second_filter_size=$s2 \
  --third_filter_size=$s3
> $dirname/mp.log 2>&1
