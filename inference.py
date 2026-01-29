import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import torch.distributed as dist
import os
from data.pairset import TARGET_CHANNELS
from data.pairset import GFS2ERA5Dataset
from models.base import G2E
from torch.utils.data import DataLoader, DistributedSampler
import multiprocessing as mp
import numpy as np
import xarray as xr
import zarr
import pandas as pd

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass
torch.backends.cudnn.deterministic = False   # 允许选择最优算法
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def inference(checkpoint_path, device, save_path, test_loader, gfs_path = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/database/gfs_2020_2024_c70_normalized"):
    print("load G2E")
    model = G2E(
        img_size=(721, 1440),
        patch_size=(4, 4),
        in_chans=70,
        embed_dim=1536, 
        depth = 4, 
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    print("model loaded")

    model.eval()
    pbar = tqdm(test_loader)
    
    preds = []
    print("inferencing...")
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(pbar):
            x = x.to(device)
            out, _, _ = model(x)
            preds.append(out.cpu().numpy())
    arr = np.concatenate(preds, axis=0)
    print("saving as zarr")
    ds_gfs = xr.open_zarr(gfs_path)


    time_list = test_loader.dataset.time_list
    ds_gfs_sel = ds_gfs.sel(time=pd.to_datetime(time_list))

    assert list(ds_gfs_sel.channel.values) == list(TARGET_CHANNELS), "channel顺序不一致"
    assert arr.shape == ds_gfs_sel["data"].shape, f"shape不一致: {arr.shape} vs {ds_gfs_sel['data'].shape}"

    ds_gfs_sel["data"].values[:] = arr

    if os.path.exists(save_path):
        import shutil
        shutil.rmtree(save_path)
    ds_gfs_sel.to_zarr(save_path, consolidated=True)
    print(f"推理结果已保存为zarr: {save_path}")

if __name__ == "__main__":

    test_set = GFS2ERA5Dataset(
        start = "2022-01-01 00:00:00",
        end = "2022-01-01 18:00:00"
    )

    test_loader = DataLoader(
        test_set,
        batch_size=8,
        shuffle=False,  
        num_workers=3,  
        pin_memory=True, 
        drop_last=False,  
    )
    print("loaded")

    inference(
        checkpoint_path = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/G2E/checkpoints/baseline_1_25/checkpoint_epoch_32.pth",
        device = "cuda",
        save_path = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/infertest/transformed_gfs",
        test_loader = test_loader,
    )