"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at the main github page.
Authors: Seung Wook Kim, Jonah Philion, Antonio Torralba, Sanja Fidler
"""

import os
import sys
import torch
import time
sys.path.append('..')
import config
import utils, visual_utils
from trainer import Trainer
import torchvision.utils as vutils
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, './data')
import dataloader
import copy
import shutil
from tqdm import tqdm
import numpy as np
import wandb
import shutil
import torchvision

from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

def setup_multip(opts):
    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opts.distributed = n_gpu > 1
    if opts.distributed:
        print('set distributed')
        torch.cuda.set_device(opts.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

def train_gamegan(gpu, opts):
    opts.device = 'cuda' if opts.gpu >= 0 else 'cpu'
    torch.backends.cudnn.benchmark = True

    opts = copy.deepcopy(opts)
    start_epoch = 0
    opts.img_size = (opts.img_size, opts.img_size)
    warm_up = opts.warm_up
    opts.gpu = gpu
    opts.width_mul = 3.5 if not utils.check_arg(opts, 'width_mul') else opts.width_mul
    opts.vqvae = utils.check_arg(opts, 'vqvae')
    opts.spatial_dim = 4 if not utils.check_arg(opts, 'spatial_dim') else opts.spatial_dim

    opts.do_latent = True
    latent_decoder = utils.get_latent_decoder(opts).to(opts.device)
    latent_decoder.eval()

    torch.manual_seed(opts.seed)

    # dataset ---
    print('setting up dataset')


    encoding_loader = dataloader.get_custom_dataset(opts, set_type=2, getLoader=True, num_workers=0, dataset_type='encoding')


    # create model
    netG, _ = utils.build_models(opts)

    checkpoint = torch.load('/mnt/DriveGAN_code/logs/carla/model200.pt')


        

    state_dict_new = checkpoint['netG']
    state_dict_old = netG.state_dict()
    for key, _ in state_dict_old.items():
        if state_dict_old[key].shape != state_dict_new[key].shape:
            print('different', key, state_dict_new[key].shape, state_dict_old[key].shape)
            state_dict_new[key] = state_dict_old[key]

    netG.load_state_dict(state_dict_new)

    


    # set up logger and trainer
    logging = True if get_rank() == 0 else False
    if logging:
        logger = SummaryWriter(opts.log_dir)


    def sample_data(loader):
        while True:
            for batch in loader:
                yield batch
    encoding_iter = sample_data(encoding_loader)
    encoding_len = len(encoding_loader)


    for step in tqdm(range(opts.num_trajectories)):
        states, actions, neg_actions, action_paths, bev_paths, measurements_paths = utils.get_data(encoding_iter, opts)

        states = [st.float() for st in states]
        actions = [ac.float() for ac in actions]

        gout = netG(states, actions, opts.min_warmup, train=False, epoch=1, latent_decoder=latent_decoder)

        vis_st = [states[0][0:1]]
        for st in gout['outputs']:
            vis_st.append(st[0:1])

        
        x_gen = torch.cat(vis_st, dim=0)
        
        x_gen = utils.run_latent_decoder(latent_decoder, x_gen, opts=opts)

        x_gen = visual_utils.rescale(x_gen)
        x_gen = torch.clamp(x_gen, 0, 1.0)

        parent_path = opts.output_path

        bev_folder = os.path.join(parent_path, str(step), 'bev')
        os.makedirs(bev_folder, exist_ok=True)
        
        measurements_folder = os.path.join(parent_path, str(step), 'measurements')
        os.makedirs(measurements_folder, exist_ok=True)

        rgb_folder = os.path.join(parent_path, str(step), 'rgb')
        os.makedirs(rgb_folder, exist_ok=True)

        supervision_folder = os.path.join(parent_path, str(step), 'supervision')
        os.makedirs(supervision_folder, exist_ok=True)


        for idx in range(x_gen.shape[0]):
            name = '0000'[:-len(str(idx))] + str(idx)

            bev_src = bev_paths[idx][0]
            shutil.copyfile(bev_src, os.path.join(bev_folder, name+'.png'))

            measurements_src = measurements_paths[idx][0]
            shutil.copyfile(measurements_src, os.path.join(measurements_folder, name+'.json'))

            torchvision.utils.save_image(x_gen[idx].cpu(), os.path.join(rgb_folder, name+'.png'))
            print(os.path.join(rgb_folder, name+'.png'))

            supervision_src = action_paths[idx][0]
            shutil.copyfile(supervision_src, os.path.join(supervision_folder, name+'.npy'))





if __name__ == '__main__':

    parser = config.init_parser()
    opts, args = parser.parse_args(sys.argv)

    train_gamegan(opts.gpu, opts)
