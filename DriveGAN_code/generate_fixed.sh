cd mnt/DriveGAN_code/

rm -r /mnt/masterproject_SoSe2023/data/generated_train/*

rm -r /mnt/masterproject_SoSe2023/data/generated_val/*

./scripts/enc_fixed.sh none /mnt/DriveGAN_code/latent_decoder_model/logs/vaegan_kernel3/160000.pt