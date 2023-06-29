import argparse
import os
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.distributions import Beta


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger

from model import TCP
from data import CARLA_Data
from config import GlobalConfig
import wandb
from train import TCP_planner



if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--checkpoint', type=str, default='')
	parser.add_argument('--root_dir_val', type=str, default='')
	parser.add_argument('--batch_size', type=int, default=32, help='Batch size')


	args = parser.parse_args()

	# Config
	config = GlobalConfig()

	# Data
	val_set = CARLA_Data(root=args.root_dir_val, data_folders=args.root_dir_val)
	print(len(val_set))

	dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

	TCP_model = TCP_planner.load_from_checkpoint(args.checkpoint, config=config, lr=0.1)
	TCP_model.eval()

	trainer = pl.Trainer.from_argparse_args(args,
											gpus = 1,
											accelerator='ddp',
											sync_batchnorm=True,
											plugins=DDPPlugin(find_unused_parameters=False),
											profiler='simple',
											benchmark=True,
											log_every_n_steps=1,
											flush_logs_every_n_steps=5,
											)

	trainer.test(TCP_model, dataloader_val)
