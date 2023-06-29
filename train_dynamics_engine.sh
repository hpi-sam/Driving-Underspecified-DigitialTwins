#!/bin/sh

#enroot start --rw --mount /hpi/fs00/share/fg-giese/masterproject_SoSe2023:mnt --mount ~:/hpi/fs00/home/philipp.hildebrandt MP /mnt/DriveGAN_code/train.sh
sleep 60m

enroot start --rw --mount /hpi/fs00/share/fg-giese/masterproject_SoSe2023:mnt --mount ~:/hpi/fs00/home/philipp.hildebrandt MP /mnt/DriveGAN_code/generate.sh