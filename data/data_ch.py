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

def _default_norm_cache_path(gfs_reader, gfs_vars: List[str], base_layers: int, pad_mode: str) -> Path:
    start_str = gfs_reader.start_dt.strftime("%Y%m%d%H")
    end_str = gfs_reader.end_dt.strftime("%Y%m%d%H")
    fname = f"gfs_norm_{start_str}_{end_str}_L{base_layers}_{pad_mode}_V{len(gfs_vars)}.npz"
    # 使用新的标准化路径
    cache_dir = Path("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/database/gfs_2020_2024_c10/gfs_norm_2020_2024.npz")
    return cache_dir / fname


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
        # base_layers: int = 13,  <-- 这个参数不再需要强行指定batch堆叠逻辑了
        # pad_mode: str = "repeat", <-- 也不再需要填充地面变量
        normalize: bool = False,
        norm_cache_path: Optional[str] = None,
        eps: float = 1e-8,
        temporal_pair: bool = False,
    ):
        self.gfs_reader = gfs_reader
        self.era5_reader = era5_reader
        # 原始的高级变量列表 (如 ['Temperature', '2 metre temperature'])
        self.raw_gfs_vars = gfs_vars or list(getattr(gfs_reader, "all_gfs_vars"))
        
        self.normalize = normalize
        self.eps = eps
        self.temporal_pair = temporal_pair

        # -----------------------------------------------------------
        # 新增逻辑：构建展开后的通道列表 (FLATTENED CHANNELS)
        # -----------------------------------------------------------
        self.flat_var_names = []
        self.var_info = [] # 记录 (原始变量名, 层索引/None, 是高空还是地面)
        
        # 预定义层级顺序，用于后缀命名
        level_suffixes = [str(l) for l in self.gfs_reader.level_order] # ['50', '100', ...]

        for var in self.raw_gfs_vars:
            var_type = self.gfs_reader.gfs_var_type.get(var, "surface")
            
            if var_type == "upper_air":
                # 对于高空变量，展开为13个通道
                for i, suffix in enumerate(level_suffixes):
                    # 例如: Temperature -> Temperature_50
                    new_name = f"{var}_{suffix}"
                    self.flat_var_names.append(new_name)
                    self.var_info.append({
                        "raw_name": var, 
                        "type": "upper_air",
                        "level_idx": i,
                        "is_separate": False # GFS读取时是整体读取
                    })
            else:
                # 地面变量保持原样
                self.flat_var_names.append(var)
                self.var_info.append({
                    "raw_name": var,
                    "type": "surface",
                    "level_idx": 0, # 地面变量通常只有一层
                    "is_separate": True 
                })
        
        # 现在的通道总数
        self.total_channels = len(self.flat_var_names)
        # print(f"数据集初始化模式：通道展开 (Flatten Mode)。总通道数：{self.total_channels}")

        # ...existing code (timestamps intersection logic)...
        gfs_times = set(gfs_reader.time_index)
        era5_times = set(era5_reader.time_index)
        self.common_timestamps = sorted(list(gfs_times & era5_times))
        if len(self.common_timestamps) == 0:
            raise ValueError("GFS 和 ERA5 没有重叠时间戳，无法配对")

        if self.temporal_pair and len(self.common_timestamps) < 2:
            raise ValueError("时间对模式需要至少两个连续时间戳")

        # 加载归一化参数逻辑微调：
        # 注意：这里假设 norm_npz 存储的还是原始变量名 (如 Temperature) 的 mean/std (shape=13)
        # 我们需要在 getitem 里切片读取对应的 mean/std
        self.norm_params: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
        if self.normalize:
            # 注意：这里的 base_layers=13 只是为了生成文件名匹配之前的缓存，逻辑上不需要了
            cache_path = Path(norm_cache_path) if norm_cache_path else _default_norm_cache_path(
                gfs_reader=self.gfs_reader,
                gfs_vars=self.raw_gfs_vars, # 用原始变量名生成 hash/filename
                base_layers=13, # 保持兼容旧文件名
                pad_mode="repeat", # 保持兼容旧文件名
            )

            if cache_path.exists():
                self.norm_params, _ = _load_norm_npz(cache_path)
            else:
                print(f"需要标准化文件: {cache_path}")
                # return (这里应该抛出错误或者生成，为了安全先不做修改)
    def __len__(self):
        """数据集长度：单时刻=时间戳数量；时间对模式=时间戳数量-1"""
        if self.temporal_pair:
            return max(0, len(self.common_timestamps) - 1)
        return len(self.common_timestamps)

    def _get_flattened_data(self, reader, ts, is_gfs=True):
        """辅助函数：读取一次时间步的所有变量并展开为 flat channels"""
        # 1. 批量读取原始数据 (返回的字典里，高空是 (13,H,W), 地面是 (1,H,W))
        # 注意 ERA5Reader 返回的也是 (13,H,W) 堆叠好的
        raw_dict = reader.read_by_time(ts, self.raw_gfs_vars, verbose=False)
        
        extracted_channels = []
        
        for info in self.var_info:
            raw_name = info["raw_name"]
            data = raw_dict[raw_name] # (D, H, W)
            
            if info["type"] == "upper_air":
                # 取出对应的那一层 (H,W)
                layer_idx = info["level_idx"]
                # data[layer_idx] 是 (H, W)，我们需要扩充一个通道维 -> (1, H, W)
                channel_data = data[layer_idx:layer_idx+1, :, :] 
                
                # 归一化逻辑
                if self.normalize:
                    # 获取该变量整体的 mean/std (13, 1, 1)
                    mean_all, std_all = self.norm_params[raw_name]
                    # 取出对应层的 mean/std
                    m = mean_all[layer_idx]
                    s = std_all[layer_idx]
                    channel_data = (channel_data - m) / s
                    
            else:
                # 地面变量 (1, H, W)
                channel_data = data 
                if self.normalize:
                    mean_all, std_all = self.norm_params[raw_name]
                    # 地面变量 mean/std 也是因为之前的 pad_to_base_layers 变成了 13层重复? 
                    # 需要检查 norm 文件生成逻辑。通常 _load_norm_npz 读取的是原始 shape
                    # 如果之前存储时已经是 shape=(13,)，那地面变量取第0个即可
                    m = mean_all[0]
                    s = std_all[0]
                    channel_data = (channel_data - m) / s

            extracted_channels.append(channel_data)
        
        # 拼接到一起: List[(1,H,W), (1,H,W)...] -> (C_total, H, W)
        return np.concatenate(extracted_channels, axis=0)

    def __getitem__(self, idx):
        if self.temporal_pair:
            ts_prev = self.common_timestamps[idx]
            ts_curr = self.common_timestamps[idx + 1]
            
            # 分别获取，形状均为 (C, H, W)
            g_prev = self._get_flattened_data(self.gfs_reader, ts_prev, is_gfs=True)
            g_curr = self._get_flattened_data(self.gfs_reader, ts_curr, is_gfs=True)
            e_prev = self._get_flattened_data(self.era5_reader, ts_prev, is_gfs=False)
            e_curr = self._get_flattened_data(self.era5_reader, ts_curr, is_gfs=False)
            
            # 堆叠时间维度 -> (C, 2, H, W)
            g_pair = np.stack([g_prev, g_curr], axis=1)
            e_pair = np.stack([e_prev, e_curr], axis=1)
            
            return torch.from_numpy(g_pair).float(), torch.from_numpy(e_pair).float(), (ts_prev.strftime("%Y-%m-%d %H:%M:%S"), ts_curr.strftime("%Y-%m-%d %H:%M:%S"))

        else:
            ts = self.common_timestamps[idx]
            g_data = self._get_flattened_data(self.gfs_reader, ts, is_gfs=True) # (C, H, W)
            e_data = self._get_flattened_data(self.era5_reader, ts, is_gfs=False) # (C, H, W)
            
            return torch.from_numpy(g_data).float(), torch.from_numpy(e_data).float(), ts.strftime("%Y-%m-%d %H:%M:%S")


def collate_fn(batch):
    # 这里的 batch 是 List[ (Tensor, Tensor, Timestamp) ]
    # 现在 Tensor 已经是展开通道的了
    
    g_list, e_list, ts_list = [], [], []
    for g, e, ts in batch:
        g_list.append(g)
        e_list.append(e)
        ts_list.append(ts)

    # 简单堆叠即可
    g_batch = torch.stack(g_list, dim=0) # (B, C, [T], H, W)
    e_batch = torch.stack(e_list, dim=0) # (B, C, [T], H, W)
    
    return g_batch, e_batch, ts_list


def validate_normalized_ranges():
    """检查标准化后的 GFS 和 ERA5 的数据范围"""
    norm_cache_path = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/database/gfs_2020_2024_c10/gfs_norm_2020_2024.npz"
    
    gfs_reader = GFSReader(
        start_dt="2020-01-01 00:00:00",
        end_dt="2020-01-05 00:00:00"
    )
    era5_reader = ERA5Reader(
        start_dt="2020-01-01 00:00:00",
        end_dt="2020-01-05 00:00:00"
    )
    
    # 创建数据集（启用标准化）
    dataset = GFSERA5PairDataset(
        gfs_reader=gfs_reader,
        era5_reader=era5_reader,
        base_layers=13,
        pad_mode="repeat",
        normalize=True,
        norm_cache_path=norm_cache_path,
        temporal_pair=False
    )
    
    print("=" * 70)
    print("标准化后的 GFS 和 ERA5 数据范围检查")
    print("=" * 70)
    
    # 采样多个时间戳来统计
    num_samples = min(5, len(dataset))
    gfs_ranges = {var: {"min": float('inf'), "max": float('-inf')} for var in dataset.gfs_vars}
    era5_ranges = {var: {"min": float('inf'), "max": float('-inf')} for var in dataset.gfs_vars}
    
    for idx in range(num_samples):
        gfs, era5, ts = dataset[idx]
        # gfs, era5 形状: (L, V, H, W)，其中 V 是变量维度
        print(f"\n时间戳 {idx + 1}/{num_samples}: {ts}")
        print(f"  GFS  数据形状: {gfs.shape}")
        print(f"  ERA5 数据形状: {era5.shape}")
        
        for v_idx, var in enumerate(dataset.gfs_vars):
            gfs_var_data = gfs[:, v_idx, :, :].numpy()  # (L, H, W)
            era5_var_data = era5[:, v_idx, :, :].numpy()
            
            gfs_min, gfs_max = gfs_var_data.min(), gfs_var_data.max()
            era5_min, era5_max = era5_var_data.min(), era5_var_data.max()
            
            gfs_ranges[var]["min"] = min(gfs_ranges[var]["min"], gfs_min)
            gfs_ranges[var]["max"] = max(gfs_ranges[var]["max"], gfs_max)
            era5_ranges[var]["min"] = min(era5_ranges[var]["min"], era5_min)
            era5_ranges[var]["max"] = max(era5_ranges[var]["max"], era5_max)
            
            print(f"    {var}:")
            print(f"      GFS  范围: [{gfs_min:.4f}, {gfs_max:.4f}]")
            print(f"      ERA5 范围: [{era5_min:.4f}, {era5_max:.4f}]")
    
    print("\n" + "=" * 70)
    print(f"汇总统计（{num_samples}个样本）:")
    print("=" * 70)
    for var in dataset.gfs_vars:
        print(f"{var}:")
        print(f"  GFS  范围: [{gfs_ranges[var]['min']:.4f}, {gfs_ranges[var]['max']:.4f}]")
        print(f"  ERA5 范围: [{era5_ranges[var]['min']:.4f}, {era5_ranges[var]['max']:.4f}]")


if __name__ == "__main__":

    gfs_reader = GFSReader(
        start_dt="2020-01-01 00:00:00",
        end_dt="2020-01-02 00:00:00",
    )
    era5_reader = ERA5Reader(
        start_dt="2020-01-01 00:00:00",
        end_dt="2020-01-02 00:00:00",
    )

    dataset = GFSERA5PairDataset(
        gfs_reader=gfs_reader,
        era5_reader=era5_reader,
        gfs_vars=None,       # 使用默认全部变量
        normalize=False,     # 先不做归一化，只看原始形状
        temporal_pair=False, # 单时刻配对
    )

    print("========== 数据集基本信息 ==========")
    print(f"样本总数: {len(dataset)}")
    print(f"展开后的总通道数 C: {dataset.total_channels}")
    print(f"展开后的通道名示例(前20个): {dataset.flat_var_names[:20]}")

    # 单条样本检查
    g, e, ts = dataset[0]
    print("\n========== 单样本检查 ==========")
    print(f"时间戳: {ts}")
    print(f"GFS 样本形状: {g.shape}  (期望: (C, H, W))")
    print(f"ERA5 样本形状: {e.shape} (期望: (C, H, W))")
    assert g.shape == e.shape, "❌ GFS 与 ERA5 的单样本形状不一致！"

    C, H, W = g.shape
    print(f"解析: C={C}, H={H}, W={W}")

    # DataLoader + collate_fn 检查
    print("\n========== DataLoader 批量检查 ==========")
    dl = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    g_batch, e_batch, ts_batch = next(iter(dl))
    print(f"批量 GFS 形状: {g_batch.shape}  (期望: (B, C, H, W))")
    print(f"批量 ERA5 形状: {e_batch.shape} (期望: (B, C, H, W))")
    assert g_batch.shape == e_batch.shape, "❌ GFS 与 ERA5 的 batch 形状不一致！"

    print(f"时间戳批量示例: {ts_batch}")

    
    