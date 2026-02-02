import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import torch.distributed as dist
import os
from .train import BaseTrainer


class DATrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def train_one_epoch(self, epoch):
        self.model.train()