#nohup python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --pool_size 50 --no_dropout
nohup python train.py --dataroot /home/ycren/python/EVUP_part --name EVUP --model cycle_gan --load_size 640 --crop_size 640 --batch_size 1 --no_html --netG SE_ResNet_blocks > /home/ycren/python/pytorch-CycleGAN-and-pix2pix/scripts/se.log 2>&1
