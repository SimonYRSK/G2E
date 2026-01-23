import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from datetime import datetime
import time
import tqdm

import multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

TARGET_CHANNELS = [
    # 温度（高空13层 + 地表t2m）
    "t50", "t100", "t150", "t200", "t250", "t300", "t400", "t500", 
    "t600", "t700", "t850", "t925", "t1000", "t2m",
    # U风（地表u10m + 高空13层）
    "u10m", "u50", "u100", "u150", "u200", "u250", "u300", "u400", 
    "u500", "u600", "u700", "u850", "u925", "u1000",
    # V风（地表v10m + 高空13层）
    "v10m", "v50", "v100", "v150", "v200", "v250", "v300", "v400", 
    "v500", "v600", "v700", "v850", "v925", "v1000",
    # 位势高度（高空13层）
    "z50", "z100", "z150", "z200", "z250", "z300", "z400", "z500", 
    "z600", "z700", "z850", "z925", "z1000",
    # 比湿（高空13层）
    "q50", "q100", "q150", "q200", "q250", "q300", "q400", "q500", 
    "q600", "q700", "q850", "q925", "q1000",
    # 地表变量
    "msl", "tp"
]

START_TIME = "2020-01-01 00:00:00"
END_TIME = "2024-12-31 00:00:00"

PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/fanjiang/dataset/era5.2002_2024.c85.p25.h6"

class GFS2ERA5Dataset(Dataset):
    def __init__(self, target_channels = None, start: str = None, end: str = None, x_path: str = None, y_path: str = None):
        self.x_path = PATH if x_path is None else x_path
        self.y_path = PATH if y_path is None else y_path
        self.target_channels = TARGET_CHANNELS if target_channels is None else target_channels


        self.start_time = pd.to_datetime(START_TIME if start is None else start)
        self.end_time = pd.to_datetime(END_TIME if end is None else end)


        self.ds_x = xr.open_zarr(self.x_path)
        self.ds_y = xr.open_zarr(self.y_path)

        x_times = pd.DatetimeIndex(self.ds_x.time.values)
        y_times = pd.DatetimeIndex(self.ds_y.time.values)

        x_times_in_range = x_times[(x_times >= self.start_time) & (x_times <= self.end_time)]
        y_times_in_range = y_times[(y_times >= self.start_time) & (y_times <= self.end_time)]

        common_times = x_times_in_range.intersection(y_times_in_range)
        self.time_list = common_times.tolist()  


        self.align_ch()


        self.lat_size = len(self.ds_x["lat"])
        self.lon_size = len(self.ds_x["lon"])
        self.chan_size = len(self.target_channels)

 

    def align_ch(self):

        self.x_all_channels = [str(c).strip() for c in self.ds_x["channel"].values]
        self.y_all_channels = [str(c).strip() for c in self.ds_y["channel"].values]

        self.x_c_idx = {name: idx for idx, name in enumerate(self.x_all_channels)}
        self.y_c_idx = {name: idx for idx, name in enumerate(self.y_all_channels)}


        self.x_target_idx = []
        self.y_target_idx = []
        for ch in self.target_channels:
            self.x_target_idx.append(self.x_c_idx[ch])
            self.y_target_idx.append(self.y_c_idx[ch])



    def __len__(self):
        return len(self.time_list)

    def __getitem__(self, idx):

        current_time = self.time_list[idx]
        
        x_data = self.ds_x["data"].sel(time=current_time).isel(channel=self.x_target_idx)
        y_data = self.ds_y["data"].sel(time=current_time).isel(channel=self.y_target_idx)

        x_np = x_data.values.astype(np.float32)
        y_np = y_data.values.astype(np.float32)
        
        x_tensor = torch.from_numpy(x_np)
        y_tensor = torch.from_numpy(y_np)

        return x_tensor, y_tensor

