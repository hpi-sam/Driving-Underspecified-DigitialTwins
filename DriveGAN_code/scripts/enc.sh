
src_dir=$1
vae_path=$2

# python generate_TCP_dataset.py \
#     --latent_decoder_model_path ${vae_path} \
#     --log_dir logs/carla \
#     --save_epoch 30 \
#     --num_gpu 1 \
#     --img_size 256 \
#     --num_steps 32 \
#     --warm_up 18 \
#     --bs 1 \
#     --hidden_dim 1792 \
#     --recon_loss_multiplier 0.1 \
#     --nfilterD_temp 32 \
#     --LAMBDA_temporal 1.0 \
#     --continuous_action True \
#     --gen_content_loss_multiplier 1.5 \
#     --lstm_num_layer 4 \
#     --eval_epoch 10 \
#     --disentangle_style True \
#     --latent_z_size 3712 \
#     --convLSTM_hidden_dim 128 \
#     --warmup_decay_step 90000 \
#     --content_kl_beta 0.1 \
#     --style_kl_beta 1.0 \
#     --theme_kl_beta 1.0 \
#     --separate_holistic_style_dim 128 \
#     --data carla_latent:${src_dir} \
#     --action_space 4 \
#     --data_path /mnt/data/train_towns_01-02-03-05 \
#     --output_path /mnt/data/generated_train \
#     --num_trajectories 2000

python generate_TCP_dataset.py \
    --latent_decoder_model_path ${vae_path} \
    --log_dir logs/carla \
    --save_epoch 30 \
    --num_gpu 1 \
    --img_size 256 \
    --num_steps 32 \
    --warm_up 18 \
    --bs 1 \
    --hidden_dim 1792 \
    --recon_loss_multiplier 0.1 \
    --nfilterD_temp 32 \
    --LAMBDA_temporal 1.0 \
    --continuous_action True \
    --gen_content_loss_multiplier 1.5 \
    --lstm_num_layer 4 \
    --eval_epoch 10 \
    --disentangle_style True \
    --latent_z_size 3712 \
    --convLSTM_hidden_dim 128 \
    --warmup_decay_step 90000 \
    --content_kl_beta 0.1 \
    --style_kl_beta 1.0 \
    --theme_kl_beta 1.0 \
    --separate_holistic_style_dim 128 \
    --data carla_latent:${src_dir} \
    --action_space 4 \
    --data_path /mnt/data/val_towns_04 \
    --output_path /mnt/data/generated_val \
    --num_trajectories 100

