import zarr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr


gfs2era5_mapping = {
    "Temperature": {
        "era5_vars": ["t50", "t100", "t150", "t200", "t250", "t300", "t400", "t500", "t600", "t700", "t850", "t925", "t1000"],
        "var_type": "upper_air",
        "description": "GFS通用气温 → ERA5 50~1000hPa等压面气温（13层）"
    },
    "2 metre temperature": {
        "era5_vars": ["t2m"],
        "var_type": "surface",
        "description": "GFS 2米气温 → ERA5 2米气温（t2m）"
    },
    "10 metre U wind component": {
        "era5_vars": ["u10m"],
        "var_type": "surface",
        "description": "GFS 10米U风分量 → ERA5 10米纬向风（u10m）"
    },
    "100 metre U wind component": {
        "era5_vars": ["u100m"],
        "var_type": "surface",
        "description": "GFS 100米U风分量 → ERA5 100米纬向风（u100m）"
    },
    "10 metre V wind component": {
        "era5_vars": ["v10m"],
        "var_type": "surface",
        "description": "GFS 10米V风分量 → ERA5 10米经向风（v10m）"
    },
    "100 metre V wind component": {
        "era5_vars": ["v100m"],
        "var_type": "surface",
        "description": "GFS 100米V风分量 → ERA5 100米经向风（v100m）"
    },
    "U component of wind": {
        "era5_vars": ["u50", "u100", "u150", "u200", "u250", "u300", "u400", "u500", "u600", "u700", "u850", "u925", "u1000"],
        "var_type": "upper_air",
        "description": "GFS通用U风分量 → ERA5 50~1000hPa等压面纬向风（13层）"
    },
    "V component of wind": {
        "era5_vars": ["v50", "v100", "v150", "v200", "v250", "v300", "v400", "v500", "v600", "v700", "v850", "v925", "v1000"],
        "var_type": "upper_air",
        "description": "GFS通用V风分量 → ERA5 50~1000hPa等压面经向风（13层）"
    },
    "Geopotential height": {
        "era5_vars": ["z50", "z100", "z150", "z200", "z250", "z300", "z400", "z500", "z600", "z700", "z850", "z925", "z1000"],
        "var_type": "upper_air",
        "description": "GFS位势高度 → ERA5 50~1000hPa等压面位势高度（13层）"
    },
    "2 metre dewpoint temperature": {
        "era5_vars": ["d2m"],
        "var_type": "surface",
        "description": "GFS 2米露点温度 → ERA5 2米露点温度（d2m）"
    }
}

# 反向映射：ERA5变量名 → GFS通用变量名
era52gfs_mapping = {}
for gfs_var, info in gfs2era5_mapping.items():
    for era5_var in info["era5_vars"]:
        era52gfs_mapping[era5_var] = gfs_var


class ERA5Reader:
    def __init__(self,
                zarr_path: str = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/fanjiang/dataset/era5.2002_2024.c85.p25.h6",
                mapping: Dict = gfs2era5_mapping,
                start_dt: str = "2020-01-01 00:00:00",
                end_dt: str = "2024-12-31 18:00:00",
                reverse_mapping: Dict = era52gfs_mapping):
        
        self.era5_root = Path(zarr_path) if isinstance(zarr_path, str) else zarr_path
        self.mapping = mapping
        self.reverse_mapping = reverse_mapping
        
        self.start_dt = datetime.strptime(start_dt, "%Y-%m-%d %H:%M:%S")
        self.end_dt = datetime.strptime(end_dt, "%Y-%m-%d %H:%M:%S")
        self.base_time = datetime(2002, 1, 1, 0, 0, 0)
        self.time_step_hours = 6.0
        
        self.gfs2era5_vars = {k: v["era5_vars"] for k, v in mapping.items()}
        
        self.data_zarr = None
        self.channel_zarr = None
        self.lat_zarr = None
        self.lon_zarr = None
        self.valid_time_indices = None
        self.channel_name2idx = None
        self._load_zarr_handles()
        
        self.valid_timestamps = self._generate_valid_timestamps()

    def _load_zarr_handles(self):
        try:
            self.data_zarr = zarr.open(self.era5_root / "data", mode='r')
            self.channel_zarr = zarr.open(self.era5_root / "channel", mode='r')
            self.lat_zarr = zarr.open(self.era5_root / "lat", mode='r')
            self.lon_zarr = zarr.open(self.era5_root / "lon", mode='r')
            
            self.time_steps = self.data_zarr.shape[0]
            self.n_channels = self.data_zarr.shape[1]
            self.lat_size = self.data_zarr.shape[2]
            self.lon_size = self.data_zarr.shape[3]
            
            assert self.lat_size == 721 and self.lon_size == 1440, \
                f"ERA5空间维度错误：需721×1440，当前{self.lat_size}×{self.lon_size}"
            
            channel_raw = self.channel_zarr[:]
            self.channel_names = []
            for name in channel_raw:
                if isinstance(name, bytes):
                    self.channel_names.append(name.decode('utf-8'))
                elif isinstance(name, (str, np.str_)):
                    self.channel_names.append(str(name))
                else:
                    self.channel_names.append(str(name))
            
            self.channel_name2idx = {name: idx for idx, name in enumerate(self.channel_names)}
            self.valid_time_indices = self._time_filter()
            
            # print(f"✅ ERA5加载完成：")
            # print(f"   有效时间步：{len(self.valid_time_indices)}个（{self.start_dt}~{self.end_dt}）")
            # print(f"   总通道数：{self.n_channels}，空间维度：{self.lat_size}×{self.lon_size}")
        
        except Exception as e:
            raise RuntimeError(f"加载ERA5 Zarr失败：{str(e)}")

    def _time_filter(self) -> List[int]:
        delta_hours_start = (self.start_dt - self.base_time).total_seconds() / 3600
        start_idx = int(delta_hours_start // self.time_step_hours)
        
        delta_hours_end = (self.end_dt - self.base_time).total_seconds() / 3600
        end_idx = int(delta_hours_end // self.time_step_hours)
        
        start_idx = max(0, start_idx)
        end_idx = min(self.time_steps - 1, end_idx)
        
        if start_idx > end_idx:
            raise ValueError(f"ERA5无有效时间数据：{self.start_dt}~{self.end_dt}（索引{start_idx}~{end_idx}超出范围）")
        
        return list(range(start_idx, end_idx + 1))

    def _generate_valid_timestamps(self) -> List[datetime]:
        valid_timestamps = []
        for time_idx in self.valid_time_indices:
            delta_hours = time_idx * self.time_step_hours
            timestamp = self.base_time + timedelta(hours=delta_hours)
            valid_timestamps.append(timestamp)
        return valid_timestamps

    def _get_nearest_time_idx(self, target_time: datetime) -> int:
        delta_hours = (target_time - self.base_time).total_seconds() / 3600
        target_idx = int(delta_hours // self.time_step_hours)
        
        valid_indices_arr = np.array(self.valid_time_indices)
        nearest_idx = valid_indices_arr[np.argmin(np.abs(valid_indices_arr - target_idx))]
        
        if nearest_idx not in self.valid_time_indices:
            raise ValueError(f"目标时间{target_time}无对应的有效ERA5数据")
        return nearest_idx

    def read_by_time(self, target_time: datetime, gfs_vars: Optional[List[str]] = None, verbose: bool = False) -> Dict[str, np.ndarray]:
        gfs_vars = gfs_vars or list(self.gfs2era5_vars.keys())
        time_idx = self._get_nearest_time_idx(target_time)
        
        result = {}
        for gfs_var in gfs_vars:
            if gfs_var not in self.gfs2era5_vars:
                raise ValueError(f"ERA5不支持GFS通用变量：{gfs_var}")
            
            era5_vars = self.gfs2era5_vars[gfs_var]
            layer_data_list = []
            
            for era5_var in era5_vars:
                if era5_var not in self.channel_name2idx:
                    raise ValueError(f"ERA5无该通道：{era5_var}")
                chan_idx = self.channel_name2idx[era5_var]
                var_data = self.data_zarr[time_idx, chan_idx, :, :]
                layer_data_list.append(var_data)
            
            var_data_3d = np.stack(layer_data_list, axis=0)
            result[gfs_var] = var_data_3d
        
        return result

    @property
    def time_index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.valid_timestamps)

    @property
    def all_gfs_vars(self) -> List[str]:
        return list(self.gfs2era5_vars.keys())


class GFSReader:
    def __init__(self,
                zarr_path: str = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/database/gfs_2020_2024_c10",
                mapping: Dict = gfs2era5_mapping,
                start_dt: str = "2020-01-01 00:00:00",
                end_dt: str = "2024-12-31 18:00:00",
                reverse_mapping: Dict = era52gfs_mapping):
        
        self.gfs_root = Path(zarr_path) if isinstance(zarr_path, str) else zarr_path
        self.mapping = mapping
        self.reverse_mapping = reverse_mapping
        
        self.start_dt = datetime.strptime(start_dt, "%Y-%m-%d %H:%M:%S")
        self.end_dt = datetime.strptime(end_dt, "%Y-%m-%d %H:%M:%S")
        
        self.gfs_var_type = {k: v["var_type"] for k, v in mapping.items()}
        self.all_gfs_vars_list = list(mapping.keys())
        
        self.time_index = None
        self.valid_time_indices = None
        self.valid_time_stamps = None
        self.level_order = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        
        # ⚡ 关键改动：只读取元数据，不持有Dataset句柄
        self._load_metadata()
        self._filter_valid_times()

    def _load_metadata(self):
        """只读取元数据，不持有Dataset句柄"""
        try:
            with xr.open_zarr(self.gfs_root) as ds:
                # 验证核心维度
                required_dims = ["time", "lat", "lon", "level"]
                missing_dims = [d for d in required_dims if d not in ds.dims]
                if missing_dims:
                    raise ValueError(f"GFS Zarr缺失核心维度：{missing_dims}")
                
                # 缓存元数据
                self.full_time_index = pd.DatetimeIndex(ds["time"].values.astype('datetime64[s]'))
                self.level_vals = ds["level"].values.tolist()  # ⚡ 缓存层级列表
                self.lat_len = ds.dims['lat']
                self.lon_len = ds.dims['lon']
                
                # print(f"✅ GFS Zarr加载完成：路径={self.gfs_root}")
                # print(f"   GFS总时间范围：{self.full_time_index[0]} ~ {self.full_time_index[-1]}")
                # print(f"   GFS总时间步：{len(self.full_time_index)}")
                # print(f"   GFS空间维度：lat={self.lat_len}, lon={self.lon_len}")
        
        except Exception as e:
            raise RuntimeError(f"加载GFS Zarr失败：{str(e)}")

    def _filter_valid_times(self):
        mask = (self.full_time_index >= self.start_dt) & (self.full_time_index <= self.end_dt)
        self.valid_time_indices = np.where(mask)[0].tolist()
        self.valid_time_stamps = self.full_time_index[mask].tolist()
        
        if not self.valid_time_indices:
            raise ValueError(f"GFS无有效时间数据：{self.start_dt}~{self.end_dt}")
        
        self.time_index = pd.DatetimeIndex(self.valid_time_stamps)
        #print(f"✅ GFS有效时间步：{len(self.valid_time_indices)}个（{self.start_dt}~{self.end_dt}）")

    def _get_nearest_time_idx(self, target_time: datetime) -> Tuple[int, datetime]:
        time_diffs = [abs((ts - target_time).total_seconds()) for ts in self.valid_time_stamps]
        nearest_pos = np.argmin(time_diffs)
        nearest_raw_idx = self.valid_time_indices[nearest_pos]
        nearest_ts = self.valid_time_stamps[nearest_pos]
        return nearest_raw_idx, nearest_ts

    def read_by_time(self, target_time: datetime, gfs_vars: Optional[List[str]] = None, verbose: bool = False) -> Dict[str, np.ndarray]:
        """⚡ 关键优化：使用with上下文管理器，用完立即关闭"""
        gfs_vars = gfs_vars or self.all_gfs_vars_list
        time_idx, _ = self._get_nearest_time_idx(target_time)
        
        result = {}
        with xr.open_zarr(self.gfs_root) as ds:
            # ⚡ 一次性切片时间
            ds_time = ds.isel(time=time_idx)
            
            for gfs_var in gfs_vars:
                if gfs_var not in self.gfs_var_type:
                    raise ValueError(f"GFS不支持该变量：{gfs_var}，支持的变量：{self.all_gfs_vars_list}")
                
                var_type = self.gfs_var_type[gfs_var]
                var_data = ds_time[gfs_var]
                
                if var_type == "upper_air":
                    level_indices = [self.level_vals.index(level) for level in self.level_order]
                    var_arr = var_data.isel(level=level_indices).values
                else:
                    var_arr = var_data.values[np.newaxis, :, :]
                
                result[gfs_var] = np.nan_to_num(var_arr, nan=0.0)
        
        return result

    @property
    def all_gfs_vars(self) -> List[str]:
        return self.all_gfs_vars_list

    def close(self):
        pass  # 不再需要手动关闭


def pad_to_base_layers(data: np.ndarray, base_layers: int = 13, pad_mode: str = "repeat") -> np.ndarray:
    D, H, W = data.shape
    if D == base_layers:
        return data
    if D != 1:
        raise ValueError(f"变量层数必须是1或{base_layers}，当前为{D}")

    if pad_mode == "repeat":
        return np.repeat(data, base_layers, axis=0)
    elif pad_mode == "zero":
        out = np.zeros((base_layers, H, W), dtype=data.dtype)
        out[0:1] = data
        return out
    else:
        raise ValueError("pad_mode must be 'repeat' or 'zero'")

def _default_norm_cache_path(era5_reader, gfs_vars: List[str], base_layers: int, pad_mode: str) -> Path:
    start_str = era5_reader.start_dt.strftime("%Y%m%d%H")
    end_str = era5_reader.end_dt.strftime("%Y%m%d%H")
    fname = f"era5_norm_{start_str}_{end_str}_L{base_layers}_{pad_mode}_V{len(gfs_vars)}.npz"
    return Path(__file__).resolve().parent / fname


def _save_norm_npz(path: Path, params: Dict[str, Tuple[np.ndarray, np.ndarray]], meta: Dict[str, str]):
    arrays = {}
    arrays["__vars__"] = np.array(list(params.keys()), dtype=object)
    for k, v in meta.items():
        arrays[f"__meta__{k}"] = np.array(str(v), dtype=object)

    for var, (mean_L, std_L) in params.items():
        arrays[f"{var}__mean"] = mean_L.astype(np.float32)
        arrays[f"{var}__std"] = std_L.astype(np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)

def _load_norm_npz(path: Path) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, str]]:
    z = np.load(path, allow_pickle=True)
    vars_list = [str(x) for x in z["__vars__"].tolist()]
    meta = {}
    for key in z.files:
        if key.startswith("__meta__"):
            meta[key.replace("__meta__", "", 1)] = str(z[key].item())

    params = {}
    for var in vars_list:
        mean_L = z[f"{var}__mean"].astype(np.float32)
        std_L = z[f"{var}__std"].astype(np.float32)
        params[var] = (mean_L, std_L)
    return params, meta


class GFSERA5PairDataset(Dataset):
    def __init__(
        self,
        gfs_reader,
        era5_reader,
        gfs_vars=None,
        base_layers: int = 13,
        pad_mode: str = "repeat",
        normalize: bool = False,
        norm_cache_path: Optional[str] = None,
        eps: float = 1e-8,
        temporal_pair: bool = False,
    ):
        self.gfs_reader = gfs_reader
        self.era5_reader = era5_reader
        self.gfs_vars = gfs_vars or list(getattr(gfs_reader, "all_gfs_vars"))
        self.base_layers = base_layers
        self.pad_mode = pad_mode
        self.normalize = normalize
        self.eps = eps
        self.temporal_pair = temporal_pair

        gfs_times = set(gfs_reader.time_index)
        era5_times = set(era5_reader.time_index)
        self.common_timestamps = sorted(list(gfs_times & era5_times))
        if len(self.common_timestamps) == 0:
            raise ValueError("GFS 和 ERA5 没有重叠时间戳，无法配对")

        if self.temporal_pair and len(self.common_timestamps) < 2:
            raise ValueError("时间对模式需要至少两个连续时间戳")

        self.norm_params: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
        if self.normalize:
            cache_path = Path(norm_cache_path) if norm_cache_path else _default_norm_cache_path(
                era5_reader=self.era5_reader,
                gfs_vars=self.gfs_vars,
                base_layers=self.base_layers,
                pad_mode=self.pad_mode,
            )

            if cache_path.exists():
                self.norm_params, _ = _load_norm_npz(cache_path)
                #print(f"✅ 读取标准化缓存：{cache_path}")
            else:
                self.norm_params = self._compute_era5_norm_params_over_full_period()
                meta = {
                    "start_dt": self.era5_reader.start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_dt": self.era5_reader.end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "base_layers": str(self.base_layers),
                    "pad_mode": str(self.pad_mode),
                }
                _save_norm_npz(cache_path, self.norm_params, meta)
                #print(f"✅ 已保存标准化缓存：{cache_path}")

    def _compute_era5_norm_params_over_full_period(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        params = {}
        era5_times = list(self.era5_reader.time_index)
        if len(era5_times) == 0:
            raise ValueError("ERA5Reader.time_index 为空，无法统计标准化参数")

        for var in self.gfs_vars:
            sum_L = np.zeros((self.base_layers,), dtype=np.float64)
            sumsq_L = np.zeros((self.base_layers,), dtype=np.float64)
            count_per_layer = 0

            for ts in era5_times:
                e = self.era5_reader.read_by_time(ts, [var], verbose=False)[var]
                e = pad_to_base_layers(e, self.base_layers, self.pad_mode)
                e = np.nan_to_num(e, nan=0.0)

                L, H, W = e.shape
                if count_per_layer == 0:
                    count_per_layer = H * W

                flat = e.reshape(L, -1).astype(np.float64)
                sum_L += flat.sum(axis=1)
                sumsq_L += (flat * flat).sum(axis=1)

            total_count = len(era5_times) * count_per_layer
            mean_L = sum_L / total_count
            var_L = sumsq_L / total_count - mean_L * mean_L
            var_L = np.maximum(var_L, 0.0)
            std_L = np.sqrt(var_L) + self.eps

            params[var] = (mean_L.astype(np.float32), std_L.astype(np.float32))

        return params

    def _norm(self, x_LHW: np.ndarray, var: str) -> np.ndarray:
        mean_L, std_L = self.norm_params[var]
        mean = mean_L[:, None, None]
        std = std_L[:, None, None]
        return (x_LHW - mean) / std

    def __len__(self):
        if self.temporal_pair:
            return len(self.common_timestamps) - 1
        return len(self.common_timestamps)

    def __getitem__(self, idx):
        if self.temporal_pair:
            ts_prev = self.common_timestamps[idx]
            ts_curr = self.common_timestamps[idx + 1]
            # 读取两帧
            g_prev_dict = self.gfs_reader.read_by_time(ts_prev, self.gfs_vars, verbose=False)
            g_curr_dict = self.gfs_reader.read_by_time(ts_curr, self.gfs_vars, verbose=False)
            e_prev_dict = self.era5_reader.read_by_time(ts_prev, self.gfs_vars, verbose=False)
            e_curr_dict = self.era5_reader.read_by_time(ts_curr, self.gfs_vars, verbose=False)

            g_prev_arrays, g_curr_arrays = [], []
            e_prev_arrays, e_curr_arrays = [], []
            for var in self.gfs_vars:
                g_prev = pad_to_base_layers(g_prev_dict[var], self.base_layers, self.pad_mode)
                g_curr = pad_to_base_layers(g_curr_dict[var], self.base_layers, self.pad_mode)
                e_curr = pad_to_base_layers(e_curr_dict[var], self.baselayers, self.pad_mode)
                e_prev = pad_to_base_layers(e_prev_dict[var], self.base__layers, self.pad_mode)

                if self.normalize:
                    g_prev = self._norm(g_prev, var)
                    g_curr = self._norm(g_curr, var)
                    e_prev = self._norm(e_prev, var)
                    e_curr = self._norm(e_curr, var)

                g_prev_arrays.append(g_prev)  # (L,H,W)
                g_curr_arrays.append(g_curr)
                e_prev_arrays.append(e_prev)
                e_curr_arrays.append(e_curr)

            # (L,V,H,W) 
            g_prev_LVHW = np.stack(g_prev_arrays, axis=1)
            g_curr_LVHW = np.stack(g_curr_arrays, axis=1)
            e_prev_LVHW = np.stack(e_prev_arrays, axis=1)
            e_curr_LVHW = np.stack(e_curr_arrays, axis=1)

            # 堆叠时间维得到 (L, V, 2, H, W)
            g_pair_LVTHW = np.stack([g_prev_LVHW, g_curr_LVHW], axis=2)
            e_pair_LVTHW = np.stack([e_prev_LVHW, e_curr_LVHW], axis=2)

            g_pair = torch.from_numpy(g_pair_LVTHW).float()  # (L, V, 2, H, W)
            e_pair = torch.from_numpy(e_pair_LVTHW).float()  # (L, V, 2, H, W)

            return g_pair, e_pair, (ts_prev.strftime("%Y-%m-%d %H:%M:%S"), ts_curr.strftime("%Y-%m-%d %H:%M:%S"))

        # 原单帧逻辑保持不变
        ts = self.common_timestamps[idx]
        gfs_dict = self.gfs_reader.read_by_time(ts, self.gfs_vars, verbose=False)
        era5_dict = self.era5_reader.read_by_time(ts, self.gfs_vars, verbose=False)

        gfs_arrays, era5_arrays = [], []
        for var in self.gfs_vars:
            g = pad_to_base_layers(gfs_dict[var], self.base_layers, self.pad_mode)
            e = pad_to_base_layers(era5_dict[var], self.base_layers, self.pad_mode)
            if self.normalize:
                g = self._norm(g, var)
                e = self._norm(e, var)
            gfs_arrays.append(g)
            era5_arrays.append(e)

        gfs = torch.from_numpy(np.stack(gfs_arrays, axis=1)).float()
        era5 = torch.from_numpy(np.stack(era5_arrays, axis=1)).float()
        return gfs, era5, ts.strftime("%Y-%m-%d %H:%M:%S")

def collate_fn(batch, base_layers: int = 13):
    # 支持三种返回：
    # - 时间对层优先: (L,V,2,H,W)  → (B*L, V, 2, H, W)
    # - 时间对通道展平旧式: (C,2,H,W) → (B, C, 2, H, W) 兼容路径
    # - 单帧: (L,V,H,W)              → (B*L, V, H, W)
    g_list, e_list, ts_list = [], [], []
    for g, e, ts in batch:
        g_list.append(g); e_list.append(e); ts_list.append(ts)

    first = g_list[0]
    # 时间对（新）：(L,V,2,H,W)
    if first.ndim == 5 and first.shape[2] == 2:
        g_stack = torch.stack(g_list, dim=0)  # (B,L,V,2,H,W)
        e_stack = torch.stack(e_list, dim=0)
        B, L, V, T, H, W = g_stack.shape
        if L != base_layers:
            raise ValueError(f"期望 L={base_layers}, 实际 L={L}")
        g_batch = g_stack.reshape(B * L, V, T, H, W)
        e_batch = e_stack.reshape(B * L, V, T, H, W)
        ts_out = []
        for ts in ts_list:
            ts_out.extend([ts] * L)  # 复制 L 次
        return g_batch, e_batch, ts_out

    # 时间对（旧）：(C,2,H,W)，保持 B 维
    if first.ndim == 4 and first.shape[1] == 2:
        g_batch = torch.stack(g_list, dim=0)  # (B,C,2,H,W)
        e_batch = torch.stack(e_list, dim=0)
        return g_batch, e_batch, ts_list

    # 单帧：(L,V,H,W)
    if first.ndim == 4:
        g_stack = torch.stack(g_list, dim=0)  # (B,L,V,H,W)
        e_stack = torch.stack(e_list, dim=0)
        B, L, V, H, W = g_stack.shape
        if L != base_layers:
            raise ValueError(f"期望 L={base_layers}, 实际 L={L}")
        g_batch = g_stack.reshape(B * L, V, H, W)
        e_batch = e_stack.reshape(B * L, V, H, W)
        ts_out = []
        for ts in ts_list:
            ts_out.extend([ts] * L)
        return g_batch, e_batch, ts_out

    raise ValueError(f"不支持的样本形状: {tuple(first.shape)}")

def check_era5_units():
    """检查 ERA5 数据的单位和元数据"""
    import zarr
    
    era5_path = Path("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/fanjiang/dataset/era5.2002_2024.c85.p25.h6")
    
    print("=" * 60)
    print("检查 ERA5 Zarr 结构:")
    
    try:
        root = zarr.open(era5_path, mode='r')
        print(f"根目录内容: {list(root.keys())}")
        
        # 检查 data array 的属性
        if 'data' in root:
            data_arr = root['data']
            print(f"\ndata shape: {data_arr.shape}")
            print(f"data dtype: {data_arr.dtype}")
            print(f"data attrs: {dict(data_arr.attrs)}")
        
        # 检查 channel 信息
        if 'channel' in root:
            channel_arr = root['channel']
            channels = channel_arr[:]
            channel_names = []
            for name in channels:
                if isinstance(name, bytes):
                    channel_names.append(name.decode('utf-8'))
                else:
                    channel_names.append(str(name))
            
            print(f"\n通道数量: {len(channel_names)}")
            print(f"前10个通道: {channel_names[:10]}")
            
            # 找温度相关的通道
            temp_channels = [c for c in channel_names if 't' in c.lower()]
            print(f"\n温度相关通道: {temp_channels[:20]}")
        
        # 检查是否有单独的 units 或 metadata
        for key in ['units', 'metadata', 'attrs', 'variables']:
            if key in root:
                print(f"\n{key}: {root[key][:]}")
        
        # 采样实际数据查看数值范围
        print("\n" + "=" * 60)
        print("采样数据数值范围:")
        
        era5_reader = ERA5Reader(
            start_dt="2020-01-01 00:00:00",
            end_dt="2020-01-01 06:00:00"
        )
        
        # 检查几个温度通道
        temp_vars = ["Temperature", "2 metre temperature"]
        ts = era5_reader.time_index[0]
        
        for var in temp_vars:
            try:
                data = era5_reader.read_by_time(ts, [var])
                arr = data[var]
                print(f"\n{var}:")
                print(f"  Shape: {arr.shape}")
                print(f"  Range: [{arr.min():.4f}, {arr.max():.4f}]")
                print(f"  Mean: {arr.mean():.4f}, Std: {arr.std():.4f}")
                print(f"  Sample values: {arr[0, 100:105, 100]}")
            except Exception as e:
                print(f"  无法读取 {var}: {e}")
                
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

def validate_alignment():
    """验证 GFS 和 ERA5 的时间和空间对齐"""
    gfs_reader = GFSReader(
        start_dt="2020-01-01 00:00:00",
        end_dt="2020-01-05 00:00:00"
    )
    era5_reader = ERA5Reader(
        start_dt="2020-01-01 00:00:00",
        end_dt="2020-01-05 00:00:00"
    )
    
    print("=" * 60)
    print("空间维度检查:")
    print(f"GFS  空间：{gfs_reader.lat_len} × {gfs_reader.lon_len}")
    print(f"ERA5 空间：{era5_reader.lat_size} × {era5_reader.lon_size}")
    if gfs_reader.lat_len == era5_reader.lat_size and gfs_reader.lon_len == era5_reader.lon_size:
        print("✅ 空间对齐")
    else:
        print("❌ 空间不对齐!")
    
    print("\n" + "=" * 60)
    print("时间范围检查:")
    print(f"GFS  时间范围：{gfs_reader.time_index[0]} ~ {gfs_reader.time_index[-1]}")
    print(f"ERA5 时间范围：{era5_reader.time_index[0]} ~ {era5_reader.time_index[-1]}")
    
    print("\n" + "=" * 60)
    print("共同时间戳检查:")
    gfs_times = set(gfs_reader.time_index)
    era5_times = set(era5_reader.time_index)
    common = sorted(list(gfs_times & era5_times))
    print(f"GFS  时间步数：{len(gfs_reader.time_index)}")
    print(f"ERA5 时间步数：{len(era5_reader.time_index)}")
    print(f"共同时间步数：{len(common)}")
    if len(common) == 0:
        print("❌ 没有共同时间戳!")
    else:
        print(f"✅ 有 {len(common)} 个共同时间戳")
        print(f"   首个共同：{common[0]}")
        print(f"   最后共同：{common[-1]}")
    
    print("\n" + "=" * 60)
    print("数据内容检查(采样一个时间戳):")
    if common:
        ts = common[0]
        gfs_data = gfs_reader.read_by_time(ts, ["Temperature"])
        era5_data = era5_reader.read_by_time(ts, ["Temperature"])
        
        g = gfs_data["Temperature"]
        e = era5_data["Temperature"]
        print(f"GFS  Temperature shape: {g.shape}")
        print(f"ERA5 Temperature shape: {e.shape}")
        print(f"GFS  数据范围: [{g.min():.2f}, {g.max():.2f}]")
        print(f"ERA5 数据范围: [{e.min():.2f}, {e.max():.2f}]")

if __name__ == "__main__":
    check_era5_units()
    validate_alignment()

# if __name__ == "__main__":
#     import time
#     from tqdm.auto import tqdm
    
#     print("=" * 60)
#     print("开始初始化 Reader...")
#     gfs_reader = GFSReader(
#         start_dt="2020-01-01 00:00:00",
#         end_dt="2024-12-31 18:00:00"
#     )
#     era5_reader = ERA5Reader(
#         start_dt="2020-01-01 00:00:00",
#         end_dt="2024-12-31 18:00:00"
#     )
    
#     train_vars = [
#         "Temperature",
#         "2 metre temperature",
#         "10 metre U wind component",
#         "100 metre U wind component",
#         "10 metre V wind component",
#         "100 metre V wind component",
#         "U component of wind",
#         "V component of wind",
#         "Geopotential height",
#         "2 metre dewpoint temperature"
#     ]
    
#     print("=" * 60)
#     print("开始初始化数据集...")
#     dataset = GFSERA5PairDataset(
#         gfs_reader=gfs_reader,
#         era5_reader=era5_reader,
#         gfs_vars=train_vars,
#         normalize=True,
#         norm_cache_path="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/database/gfs_2020_2024_c10/era5_norm_1_8.npz",
#         base_layers=13,
#         pad_mode="repeat",
#     )
    
#     print(f"✅ 数据集初始化完成，共 {len(dataset)} 个样本")

    
#     batch_size = 2
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=0,
#         collate_fn=lambda x: collate_fn(x, base_layers=13)
#     )
#     print("dataloader加载完毕")

#     for batch_idx, (gfs_batch, era5_batch, ts_batch) in enumerate(dataloader):
#         print("gfs_batch shape:", tuple(gfs_batch.shape))   # (B*13, V, 2, 721, 1440)
#         print("era5_batch shape:", tuple(era5_batch.shape)) # (B*13, V, 2, 721, 1440)
#         print("ts pairs sample:", ts_batch[:2])
#         break
    
    