import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import torch.distributed as dist
import os
from data.pairset import TARGET_CHANNELS
from data.pairset import GFS2ERA5Dataset
from models.swinVAE import G2E
from models.vanilaVAE import G2Esimple
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
    # model = G2Esimple(
    #     img_size=(721, 1440),
    #     patch_size=(4, 4),
    #     in_chans=70,
    #     embed_dim=1024, 
    #     num_stages = 1, 
    #     using_checkpoints = False
    # ).to(device)

    model = G2E(
        img_size=(721, 1440),
        patch_size=(4, 4),
        in_chans=70,  # 匹配你的
        embed_dim=1024,  
        num_stages=1,  
        depth=2,  # 加Swin，从小depth开始
        using_checkpoints=True,
        using_time_embedding = True,
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    print("model loaded")

    model.eval()
    pbar = tqdm(test_loader)
    
    preds = []
    print("inferencing...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # 兼容 dataset 返回 (x,y) 或 (x,y,i,times)
            if isinstance(batch, (list, tuple)) and len(batch) == 4:
                x, _, i, times = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, _ = batch
                i, times = None, None
            else:
                raise ValueError(f"Unexpected batch format: type={type(batch)}, len={len(batch) if hasattr(batch, '__len__') else 'N/A'}")

            x = x.to(device)
            if i is not None and torch.is_tensor(i):
                i = i.to(device)

            # 有时间输入就传，没有就走原始分支
            if i is not None and times is not None:
                out, _, _ = model(x, i=i, times=times)
            else:
                out, _, _ = model(x)
            preds.append(out.cpu().numpy())
    arr = np.concatenate(preds, axis=0)
    print("saving as zarr")
    ds_gfs = xr.open_zarr(gfs_path)


    time_list = test_loader.dataset.time_list
    ds_gfs_sel = ds_gfs.sel(time=pd.to_datetime(time_list))

    new_ds = xr.Dataset(
        {
            "data": (("time", "channel", "lat", "lon"), arr)
        },
        coords={
            "time": ds_gfs_sel.time,
            "channel": ds_gfs_sel.channel,
            "lat": ds_gfs_sel.lat,
            "lon": ds_gfs_sel.lon,
        },
        attrs=ds_gfs_sel.attrs  # 可选，继承原有属性
    )

    assert list(ds_gfs_sel.channel.values) == list(TARGET_CHANNELS), "channel顺序不一致"
    assert arr.shape == ds_gfs_sel["data"].shape, f"shape不一致: {arr.shape} vs {ds_gfs_sel['data'].shape}"

    if os.path.exists(save_path):
        import shutil
        shutil.rmtree(save_path)
    new_ds.to_zarr(save_path, consolidated=True)
    print(f"推理结果已保存为zarr: {save_path}")

if __name__ == "__main__":

    test_set = GFS2ERA5Dataset(
        start = "2024-01-01 00:00:00",
        end = "2024-01-01 18:00:00"
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
        checkpoint_path = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/G2E/checkpoints/t-swin3_7/checkpoint_epoch_140.pth",
        device = "cuda",
        save_path = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/G2E/inferenced/t-swin3_7",
        test_loader = test_loader,
    )


    