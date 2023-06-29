#!/usr/bin/env bash

#cd mnt/DriveGAN_code/latent_decoder_model

python main.py \
    --path /mnt/data/train_towns_01-02-03-05 \
    --val_path /mnt/data/val_towns_04 \
    --batch 8 \
    --size 256 \
    --dataset carla \
    --gamma 50.0 \
    --theme_beta 1.0 \
    --spatial_beta 2.0 \
    --log_dir ./logs/pretrained_finetune \
    --ckpt /mnt/DriveGAN_code/models/vaegan_iter210000.pt

