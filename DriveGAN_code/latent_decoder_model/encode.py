"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at the main github page.
Authors: Seung Wook Kim, Jonah Philion, Antonio Torralba, Sanja Fidler
"""

import argparse
import os
import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
from torch.utils import data
from dataset import EncodingDataset
import json
from model.model import styleVAEGAN
from tqdm import tqdm
torch.backends.cudnn.benchmark = True


def make_image(tensor, logger, title, video=True, global_step=0):
    tensor = tensor * 0.5 + 0.5
    tensor = torch.clamp(tensor, 0, 1.0)

    if not video:
        tensor = utils.make_grid(
            tensor, nrow=1,
            normalize=False, scale_each=False
        )
        logger.add_image(title, tensor, global_step=global_step)
    else:
        logger.add_video(title, tensor.unsqueeze(0), global_step=global_step)

def load_data(ld, device, args):
    next_data = next(ld)
    if next_data is None:
        return None

    imgs, path = next_data
    imgs = imgs.to(device)
    imgs = imgs.squeeze(0)

    return imgs, path


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def visualize(logger, g_ema, imgs, merged_results, path, args, cur_fname=None):
    if isinstance(path, list):
        p = path[0][0]
    if isinstance(path, tuple):
        p = path[0]
    cur_dir = p.split('/')[-2]
    cur_fname = p.split('/')[-1] if cur_fname is None else cur_fname
    make_image(imgs, logger, cur_dir+'_'+cur_fname+'/GT')

    bs = merged_results['theme_mu'].size(0)
    minibatch_size = bs # set smaller if gpu mem exceeds
    collect = []
    for mind in range(int(np.ceil(bs / minibatch_size))):
        cur_gt = imgs[mind*minibatch_size:(mind+1)*minibatch_size]
        cur_theme = merged_results['theme_mu'][mind*minibatch_size:(mind+1)*minibatch_size]
        cur_spatial = merged_results['spatial_mu'].view(minibatch_size, -1, g_ema.constant_input_size, int(g_ema.constant_input_size*g_ema.args.width_mul))

        out = g_ema({'theme_z':cur_theme, 'spatial_z': cur_spatial}, decode_only=True)
        collect.append(out['image'])
    out = {'image': torch.cat(collect, dim=0)}
    make_image(out['image'], logger, cur_dir+'_'+cur_fname+'/z')


def save_data(merged_results, ind, path, key, args):
    if isinstance(path, tuple):
        path = path[0]
        path = os.path.dirname(path)
    json_path = os.path.join(path, 'info.json')

    cur_dir = path.split('/')[-2]
    cur_fname = path.split('/')[-1]

    cur_file = json.load(open(json_path, 'rb'))

    path = [p[0] for p in path]


    for k, item in merged_results.items():
        cur_file[k] = item.cpu().data.numpy()

    cur_file['paths'] = path
    cur_fname = os.path.join(args.results_path, key[0]+'.npy')
    np.save(cur_fname, cur_file)

def save_data_pilotnet(merged_results, ind, path, key, args):

    path = [p[0] for p in path]
    cur_file = {}
    for k, item in merged_results.items():
        cur_file[k] = item.cpu().data.numpy()

    cur_file['paths'] = path
    cur_fname = os.path.join(args.results_path, key[0]+'.npy')
    np.save(cur_fname, cur_file)

def encode(encoder, loader, args, device):

    num_batch = len(loader)
    print('\n\nnum_batch: ' + str(num_batch) + '\n\n')

    loader = sample_data(loader)

    viz, logger = None, None
    if args.visualize:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(args.results_path)

    for ind in tqdm(range(num_batch)):
        next_data = load_data(loader, device, args)

        imgs, path = next_data
        print('IN', path)
        path = path[0]
        imgs = imgs.unsqueeze(0)

        basepath = '/'.join(path.split('/')[:-1])
        name = (path.split('/')[-1]).split('.')[0]

        latentpath = basepath + '/' + name + '.pkl'

        if os.path.exists(latentpath):
            os.remove(latentpath)


        with torch.no_grad():
            res = encoder(imgs, return_latent_only=True)

        print('OUT', latentpath)
        torch.save(res, latentpath)


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--cur_ind', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--num_chunk', type=int, default=0)
    parser.add_argument('--num_div_batch', type=int, default=2)

    parser.add_argument('--results_path', type=str, default='/results')
    parser.add_argument('--data_path', type=str, default='/mount/data')
    parser.add_argument('--dataset', type=str, default='carla')
    parser.add_argument('--visualize', type=int, default=0)
    parser.add_argument('--width_mul', type=float, default=1.0)
    parser.add_argument('--crop_input', type=int, default=0)
    parser.add_argument('--test', type=int, default=0)
    args = parser.parse_args()
    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1

    #### load trained styleVAEGAN model
    saved_file = torch.load(args.ckpt)
    saved_args = saved_file['args']
    vae_model = styleVAEGAN
    encoder = vae_model(
        saved_args.size, n_mlp=saved_args.n_mlp, channel_multiplier=saved_args.channel_multiplier, args=saved_args,
    ).to(device)
    encoder.load_state_dict(saved_file['vae_ema'], strict=False)

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            return None
        return torch.utils.data.dataloader.default_collate(batch)

    #### define dataloader
    dataset = EncodingDataset(args.data_path, args.dataset, args.size, args=args)
    loader = data.DataLoader(
        dataset,
        batch_size=1,
        sampler=data.SequentialSampler(dataset),
        drop_last=False,
        collate_fn=collate_fn
    )

    encode(encoder, loader, args, device)
