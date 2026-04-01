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

START_TIME = "2021-01-01 00:00:00"
END_TIME = "2024-12-31 18:00:00"

ERA5_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/huangqiusheng/datasets/era5.rtm.02_25.6h.c109.new3/"
GFS_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/database/gfs_2020_2024_c70_normalized"

BAD_TIMES = {
    pd.Timestamp("202401010000"),
    pd.Timestamp("202501010000"),
}

class GFS2ERA5Dataset(Dataset):
    def __init__(
        self,
        target_channels=None,
        start: str | None = None,
        end: str | None = None,
        x_path: str | None = None,
        y_path: str | None = None,
        target_mode: str = "era5",
        # 验证集：指定年份，每个月随机抽取若干“整天”的所有时间步
        val_sample_per_month: int | None = None,
        val_sample_year: int | None = None,
        # 训练集：每个年份最多保留多少个时间步，用于快速调参
        max_samples_per_year: int | None = None,
        sample_seed: int = 42,
    ):
        self.x_path = GFS_PATH if x_path is None else x_path
        self.y_path = ERA5_PATH if y_path is None else y_path
        self.target_channels = TARGET_CHANNELS if target_channels is None else target_channels
        self.target_mode = str(target_mode).lower()
        if self.target_mode not in {"era5", "diff"}:
            raise ValueError(f"target_mode must be 'era5' or 'diff', got: {target_mode}")


        self.start_time = pd.to_datetime(START_TIME if start is None else start)
        self.end_time = pd.to_datetime(END_TIME if end is None else end)

        self.val_sample_per_month = val_sample_per_month
        self.val_sample_year = val_sample_year
        self.max_samples_per_year = max_samples_per_year
        self.sample_seed = int(sample_seed)

        # 显式使用 consolidated=False 打开 Zarr，避免自动探测失败的警告
        self.ds_x = xr.open_zarr(self.x_path, consolidated=False)
        self.ds_y = xr.open_zarr(self.y_path, consolidated=False)

        x_times = pd.DatetimeIndex(self.ds_x.time.values)
        y_times = pd.DatetimeIndex(self.ds_y.time.values)

        x_times_in_range = x_times[(x_times >= self.start_time) & (x_times <= self.end_time)]
        y_times_in_range = y_times[(y_times >= self.start_time) & (y_times <= self.end_time)]

        common_times = x_times_in_range.intersection(y_times_in_range)
        
        # 过滤掉坏时间步
        if BAD_TIMES:
            mask = ~common_times.isin(BAD_TIMES)
            common_times = common_times[mask]

        # 若为验证集：在指定年份内，每个月随机抽取若干“整天”的所有时间步
        if self.val_sample_per_month is not None and self.val_sample_year is not None:
            rng = np.random.default_rng(self.sample_seed)
            times_year = common_times[common_times.year == self.val_sample_year]

            selected_ts: list[pd.Timestamp] = []
            for month in range(1, 13):
                month_times = times_year[times_year.month == month]
                if len(month_times) == 0:
                    continue

                # 按“天”去重，随机选若干天，再保留这些天里的全部时间步
                days = month_times.normalize().unique()
                if len(days) == 0:
                    continue

                k = min(self.val_sample_per_month, len(days))
                chosen_days = rng.choice(days, size=k, replace=False)

                for d in chosen_days:
                    mask_d = month_times.normalize() == d
                    selected_ts.extend(month_times[mask_d].tolist())

            if selected_ts:
                common_times = pd.DatetimeIndex(sorted(selected_ts))

        # 若为训练集快速调参：每个年份最多保留若干时间步
        if self.max_samples_per_year is not None and self.max_samples_per_year > 0:
            rng = np.random.default_rng(self.sample_seed)
            selected_ts: list[pd.Timestamp] = []

            for year in sorted(common_times.year.unique()):
                year_times = common_times[common_times.year == year]
                n = len(year_times)
                if n <= self.max_samples_per_year:
                    selected_ts.extend(year_times.tolist())
                else:
                    idx = rng.choice(n, size=self.max_samples_per_year, replace=False)
                    selected_ts.extend(year_times.sort_values().to_series().iloc[idx].tolist())

            if selected_ts:
                common_times = pd.DatetimeIndex(sorted(selected_ts))

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

        # 目标模式：
        # - era5: 直接学习 GFS -> ERA5
        # - diff: 学习 ERA5 - GFS 的差值（无需提前生成差值 zarr）
        if self.target_mode == "diff":
            y_np = y_np - x_np
        
        x_tensor = torch.from_numpy(x_np)
        y_tensor = torch.from_numpy(y_np)

        # hour: 0~23
        hour = current_time.hour
        # day of year: 1~366
        doy = current_time.dayofyear

        hour_tensor = torch.tensor(hour, dtype=torch.long)
        doy_tensor = torch.tensor(doy, dtype=torch.long)

        return x_tensor, y_tensor, idx, str(current_time)

