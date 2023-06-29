#!/bin/sh
python server.py  \
    --saved_model /hpi/fs00/share/fg-friedrich/carla/simulator_epoch1020.pt \
    --initial_screen rand \
    --play \
    --seed 222 \
    --gpu 0 \
    --port 8888 \
    --latent_decoder_model_path /hpi/fs00/share/fg-friedrich/carla/vaegan_iter210000.pt \