#!/bin/bash
source /home/ycren/miniconda3/bin/activate
conda activate pytorch-CycleGAN-and-pix2pix
param=("SE_ResNet_blocks" "resnet_9blocks")
for item in "${param[@]}"; do
	nohup python train.py --dataroot /home/ycren/python/EVUP_part --name ${item}_cyclegan --model cycle_gan --load_size 640 --crop_size 640 --batch_size 1 --netG ${item} --no_html > /home/ycren/python/pytorch-CycleGAN-and-pix2pix/scripts/${item}.log 2>&1
done
