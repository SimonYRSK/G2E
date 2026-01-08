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
        
        # 路径处理（兼容字符串/Path对象）
        self.era5_root = Path(zarr_path) if isinstance(zarr_path, str) else zarr_path
        self.mapping = mapping
        self.reverse_mapping = reverse_mapping
        
        # 时间参数初始化
        self.start_dt = datetime.strptime(start_dt, "%Y-%m-%d %H:%M:%S")
        self.end_dt = datetime.strptime(end_dt, "%Y-%m-%d %H:%M:%S")
        self.base_time = datetime(2002, 1, 1, 0, 0, 0)  # ERA5数据集基准时间
        self.time_step_hours = 6.0  # ERA5固定6小时时间步
        
        # 预缓存：GFS通用变量名 → ERA5变量列表
        self.gfs2era5_vars = {k: v["era5_vars"] for k, v in mapping.items()}
        
        # 懒加载Zarr句柄和有效时间索引
        self.data_zarr = None
        self.channel_zarr = None
        self.lat_zarr = None
        self.lon_zarr = None
        self.valid_time_indices = None  # 筛选后的有效时间索引
        self.channel_name2idx = None    # channel名→索引映射
        self._load_zarr_handles()
        
        # 预生成有效时间戳列表（索引→实际时间）
        self.valid_timestamps = self._generate_valid_timestamps()

    def _load_zarr_handles(self):
        """懒加载Zarr句柄（核心：避免一次性加载超大数组）"""
        try:
            # 加载各维度Zarr句柄（只读模式）
            self.data_zarr = zarr.open(self.era5_root / "data", mode='r')
            self.channel_zarr = zarr.open(self.era5_root / "channel", mode='r')
            self.lat_zarr = zarr.open(self.era5_root / "lat", mode='r')
            self.lon_zarr = zarr.open(self.era5_root / "lon", mode='r')
            
            # 基础维度信息
            self.time_steps = self.data_zarr.shape[0]  # time维度长度
            self.n_channels = self.data_zarr.shape[1]  # channel维度长度
            self.lat_size = self.data_zarr.shape[2]    # 纬度数
            self.lon_size = self.data_zarr.shape[3]    # 经度数
            
            # 校验空间维度（721×1440）
            assert self.lat_size == 721 and self.lon_size == 1440, \
                f"ERA5空间维度错误：需721×1440，当前{self.lat_size}×{self.lon_size}"
            
            # 兼容字符串/字节串的channel名
            channel_raw = self.channel_zarr[:]
            self.channel_names = []
            for name in channel_raw:
                if isinstance(name, bytes):
                    self.channel_names.append(name.decode('utf-8'))
                elif isinstance(name, (str, np.str_)):
                    self.channel_names.append(str(name))
                else:
                    self.channel_names.append(str(name))
            
            # 构建channel名→索引映射
            self.channel_name2idx = {name: idx for idx, name in enumerate(self.channel_names)}
            
            # 筛选有效时间索引
            self.valid_time_indices = self._time_filter()
            print(f"✅ ERA5加载完成：")
            print(f"   有效时间步：{len(self.valid_time_indices)}个（{self.start_dt}~{self.end_dt}）")
            print(f"   总通道数：{self.n_channels}，空间维度：{self.lat_size}×{self.lon_size}")
            print(f"   前5个通道名：{self.channel_names[:5]}")
        
        except Exception as e:
            raise RuntimeError(f"加载ERA5 Zarr失败：{str(e)}")

    def _time_filter(self) -> List[int]:
        """筛选指定时间范围内的有效时间索引"""
        # 计算起始/结束时间对应的索引
        delta_hours_start = (self.start_dt - self.base_time).total_seconds() / 3600
        start_idx = int(delta_hours_start // self.time_step_hours)
        
        delta_hours_end = (self.end_dt - self.base_time).total_seconds() / 3600
        end_idx = int(delta_hours_end // self.time_step_hours)
        
        # 边界校验
        start_idx = max(0, start_idx)
        end_idx = min(self.time_steps - 1, end_idx)
        
        if start_idx > end_idx:
            raise ValueError(f"ERA5无有效时间数据：{self.start_dt}~{self.end_dt}（索引{start_idx}~{end_idx}超出范围）")
        
        return list(range(start_idx, end_idx + 1))

    def _generate_valid_timestamps(self) -> List[datetime]:
        """生成有效时间索引对应的实际时间戳列表"""
        valid_timestamps = []
        for time_idx in self.valid_time_indices:
            delta_hours = time_idx * self.time_step_hours
            timestamp = self.base_time + timedelta(hours=delta_hours)
            valid_timestamps.append(timestamp)
        return valid_timestamps

    def _get_nearest_time_idx(self, target_time: datetime) -> int:
        """根据目标时间找最近的有效时间索引"""
        delta_hours = (target_time - self.base_time).total_seconds() / 3600
        target_idx = int(delta_hours // self.time_step_hours)
        
        # 找有效索引中最接近的
        valid_indices_arr = np.array(self.valid_time_indices)
        nearest_idx = valid_indices_arr[np.argmin(np.abs(valid_indices_arr - target_idx))]
        
        if nearest_idx not in self.valid_time_indices:
            raise ValueError(f"目标时间{target_time}无对应的有效ERA5数据")
        return nearest_idx

    def read_by_time(self, target_time: datetime, gfs_vars: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        按目标时间读取ERA5数据，返回{GFS通用变量名: (层数, lat, lon)}
        """
        gfs_vars = gfs_vars or list(self.gfs2era5_vars.keys())
        
        # 1. 找到最近的有效时间索引
        time_idx = self._get_nearest_time_idx(target_time)
        
        # 2. 逐个读取GFS通用变量对应的ERA5分层变量
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
                
                # 读取数据：(time, channel, lat, lon) → 取指定time和channel
                var_data = self.data_zarr[time_idx, chan_idx, :, :]  # (lat, lon)
                layer_data_list.append(var_data)
            
            # 拼接分层变量 → (层数, lat, lon)
            var_data_3d = np.stack(layer_data_list, axis=0)
            result[gfs_var] = var_data_3d
            print(f"📌 ERA5通用变量{gfs_var}：拼接{len(era5_vars)}层，形状{var_data_3d.shape}")
        
        return result

    @property
    def time_index(self) -> pd.DatetimeIndex:
        """返回有效时间戳的DatetimeIndex（和GFSReader对齐）"""
        return pd.DatetimeIndex(self.valid_timestamps)

    @property
    def all_gfs_vars(self) -> List[str]:
        """返回所有支持的GFS通用变量名"""
        return list(self.gfs2era5_vars.keys())


class GFSReader:
    def __init__(self,
                zarr_path: str = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/database/gfs_2020_2024_c10",
                mapping: Dict = gfs2era5_mapping,
                start_dt: str = "2020-01-01 00:00:00",
                end_dt: str = "2024-12-31 18:00:00",
                reverse_mapping: Dict = era52gfs_mapping):
        
        # 路径处理
        self.gfs_root = Path(zarr_path) if isinstance(zarr_path, str) else zarr_path
        self.mapping = mapping
        self.reverse_mapping = reverse_mapping
        
        # 时间参数初始化
        self.start_dt = datetime.strptime(start_dt, "%Y-%m-%d %H:%M:%S")
        self.end_dt = datetime.strptime(end_dt, "%Y-%m-%d %H:%M:%S")
        
        # 预缓存：GFS通用变量名 → 变量类型（高空/地面）
        self.gfs_var_type = {k: v["var_type"] for k, v in mapping.items()}
        # 预缓存：所有支持的GFS变量名
        self.all_gfs_vars_list = list(mapping.keys())
        
        # 懒加载属性
        self.ds = None  # xarray Dataset句柄
        self.time_index = None  # 有效时间索引（DatetimeIndex）
        self.valid_time_indices = None  # 筛选后的原始时间索引（整数列表）
        self.valid_time_stamps = None   # 筛选后的时间戳列表
        self.level_order = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]  # 固定层级顺序
        
        # 初始化加载
        self._load_zarr_handles()
        self._filter_valid_times()

    def _load_zarr_handles(self):
        """懒加载GFS Zarr数据（xarray方式）"""
        try:
            self.ds = xr.open_zarr(self.gfs_root)
            print(f"✅ GFS Zarr加载完成：路径={self.gfs_root}")
            
            # 验证核心维度
            required_dims = ["time", "lat", "lon", "level"]
            missing_dims = [d for d in required_dims if d not in self.ds.dims]
            if missing_dims:
                raise ValueError(f"GFS Zarr缺失核心维度：{missing_dims}")
            
            # 构建完整的时间索引
            self.full_time_index = pd.DatetimeIndex(self.ds["time"].values.astype('datetime64[s]'))
            print(f"   GFS总时间范围：{self.full_time_index[0]} ~ {self.full_time_index[-1]}")
            print(f"   GFS总时间步：{len(self.full_time_index)}")
            print(f"   GFS空间维度：lat={self.ds.dims['lat']}, lon={self.ds.dims['lon']}")
        
        except Exception as e:
            raise RuntimeError(f"加载GFS Zarr失败：{str(e)}")

    def _filter_valid_times(self):
        """筛选指定时间范围内的有效时间索引和时间戳"""
        mask = (self.full_time_index >= self.start_dt) & (self.full_time_index <= self.end_dt)
        self.valid_time_indices = np.where(mask)[0].tolist()
        self.valid_time_stamps = self.full_time_index[mask].tolist()
        
        if not self.valid_time_indices:
            raise ValueError(f"GFS无有效时间数据：{self.start_dt}~{self.end_dt}")
        
        self.time_index = pd.DatetimeIndex(self.valid_time_stamps)
        print(f"✅ GFS有效时间步：{len(self.valid_time_indices)}个（{self.start_dt}~{self.end_dt}）")

    def _get_nearest_time_idx(self, target_time: datetime) -> Tuple[int, datetime]:
        """找到目标时间最近的有效时间索引和对应的时间戳"""
        time_diffs = [abs((ts - target_time).total_seconds()) for ts in self.valid_time_stamps]
        nearest_pos = np.argmin(time_diffs)
        nearest_raw_idx = self.valid_time_indices[nearest_pos]
        nearest_ts = self.valid_time_stamps[nearest_pos]
        return nearest_raw_idx, nearest_ts

    def read_by_time(self, target_time: datetime, gfs_vars: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """按目标时间读取GFS数据（和ERA5Reader接口完全对齐）"""
        gfs_vars = gfs_vars or self.all_gfs_vars_list
        
        # 1. 找到最近的有效时间索引和时间戳
        time_idx, target_time_str = self._get_nearest_time_idx(target_time)
        print(f"📌 读取GFS时间：{target_time_str}（目标时间：{target_time}）")
        
        # 2. 逐个读取变量
        result = {}
        for gfs_var in gfs_vars:
            if gfs_var not in self.gfs_var_type:
                raise ValueError(f"GFS不支持该变量：{gfs_var}，支持的变量：{self.all_gfs_vars_list}")
            
            var_type = self.gfs_var_type[gfs_var]
            var_data = self.ds[gfs_var].isel(time=time_idx)  # 取指定时间
            
            if var_type == "upper_air":
                # 高空变量：按固定层级顺序重新排序 → (13, 721, 1440)
                level_vals = self.ds["level"].values.tolist()
                try:
                    level_indices = [level_vals.index(level) for level in self.level_order]
                except ValueError as e:
                    raise ValueError(f"GFS层级缺失：{e}，当前层级：{level_vals}")
                var_data_sorted = var_data.isel(level=level_indices)
                var_arr = var_data_sorted.values  # (13, 721, 1440)
            else:
                # 地面变量：增加层数维度 → (1, 721, 1440)
                var_arr = var_data.values[np.newaxis, :, :]
            
            # 处理缺失值
            var_arr = np.nan_to_num(var_arr, nan=0.0)
            result[gfs_var] = var_arr
            print(f"   GFS变量{gfs_var}：形状{var_arr.shape}（{var_type}）")
        
        return result

    @property
    def all_gfs_vars(self) -> List[str]:
        """返回所有支持的GFS变量名"""
        return self.all_gfs_vars_list

    def close(self):
        """关闭Dataset句柄，释放资源"""
        if self.ds is not None:
            self.ds.close()
            print("✅ GFS Dataset已关闭")



import numpy as np
import torch
from torch.utils.data import Dataset

def pad_to_base_layers(data: np.ndarray, base_layers: int = 13, pad_mode: str = "repeat") -> np.ndarray:
    """
    data: (D, H, W), D in {1, base_layers}
    return: (base_layers, H, W)
    """
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

class GFSERA5PairDataset(Dataset):
    """
    单样本返回:
      gfs: (L, V, H, W)
      era5:(L, V, H, W)
      ts:  str
    其中 L=base_layers(默认13), V=len(gfs_vars)
    """
    def __init__(
        self,
        gfs_reader,
        era5_reader,
        gfs_vars=None,
        base_layers: int = 13,
        pad_mode: str = "repeat",
        normalize: bool = False,
        norm_sample_num: int = 100,
        eps: float = 1e-8,
    ):
        self.gfs_reader = gfs_reader
        self.era5_reader = era5_reader
        self.gfs_vars = gfs_vars or list(getattr(gfs_reader, "all_gfs_vars"))
        self.base_layers = base_layers
        self.pad_mode = pad_mode
        self.normalize = normalize
        self.norm_sample_num = norm_sample_num
        self.eps = eps

        # 取交集时间戳，确保严格配对（不会“各自 nearest 造成错配”）
        gfs_times = set(gfs_reader.time_index)
        era5_times = set(era5_reader.time_index)
        self.common_timestamps = sorted(list(gfs_times & era5_times))
        if len(self.common_timestamps) == 0:
            raise ValueError("GFS 和 ERA5 没有重叠时间戳，无法配对")

        # (可选) 按变量做标准化：每个变量独立 mean/std（对 pad 后的 (L,H,W) 统计）
        self.norm_params = None
        if self.normalize:
            self.norm_params = self._compute_norm_params()

    def _compute_norm_params(self):
        params = {}
        n = min(self.norm_sample_num, len(self.common_timestamps))
        for var in self.gfs_vars:
            buf = []
            for ts in self.common_timestamps[:n]:
                g = self.gfs_reader.read_by_time(ts, [var])[var]
                e = self.era5_reader.read_by_time(ts, [var])[var]
                g = pad_to_base_layers(g, self.base_layers, self.pad_mode)
                e = pad_to_base_layers(e, self.base_layers, self.pad_mode)
                buf.append(g); buf.append(e)
            x = np.stack(buf, axis=0)  # (2n, L, H, W)
            mean = np.mean(x, axis=(0, 2, 3), keepdims=True)  # (1, L, 1, 1)
            std = np.std(x, axis=(0, 2, 3), keepdims=True) + self.eps
            params[var] = (mean, std)
        return params

    def _norm(self, x: np.ndarray, var: str) -> np.ndarray:
        mean, std = self.norm_params[var]
        # x: (L,H,W) -> broadcast with (1,L,1,1) by adding axis
        return (x[None, ...] - mean) / std  # (1,L,H,W)

    def __len__(self):
        return len(self.common_timestamps)

    def __getitem__(self, idx):
        ts = self.common_timestamps[idx]
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")

        gfs_vars_LVHW = []
        era5_vars_LVHW = []

        for var in self.gfs_vars:
            g = self.gfs_reader.read_by_time(ts, [var])[var]   # (D,H,W)
            e = self.era5_reader.read_by_time(ts, [var])[var]  # (D,H,W)

            g = pad_to_base_layers(g, self.base_layers, self.pad_mode)  # (L,H,W)
            e = pad_to_base_layers(e, self.base_layers, self.pad_mode)  # (L,H,W)

            if self.normalize:
                g = self._norm(g, var)[0]  # back to (L,H,W)
                e = self._norm(e, var)[0]

            # 变成 (L,1,H,W)
            gfs_vars_LVHW.append(torch.from_numpy(g).float().unsqueeze(1))
            era5_vars_LVHW.append(torch.from_numpy(e).float().unsqueeze(1))

        # 合并变量维： (L,V,H,W)
        gfs = torch.cat(gfs_vars_LVHW, dim=1)
        era5 = torch.cat(era5_vars_LVHW, dim=1)

        return gfs, era5, ts_str

def collate_fn(batch, base_layers: int = 13):
    """
    输入batch: List[(gfs(L,V,H,W), era5(L,V,H,W), ts_str)]
    输出:
      gfs_batch: (B*L, V, H, W)
      era5_batch: (B*L, V, H, W)
      ts_batch:   List[str] 长度 B*L（每层复制时间戳）
    """
    g_list, e_list, ts_list = [], [], []
    for g, e, ts in batch:
        if g.ndim != 4 or g.shape[0] != base_layers:
            raise ValueError(f"期望单样本 gfs 为 ({base_layers},V,H,W)，实际 {tuple(g.shape)}")
        if e.ndim != 4 or e.shape[0] != base_layers:
            raise ValueError(f"期望单样本 era5 为 ({base_layers},V,H,W)，实际 {tuple(e.shape)}")
        g_list.append(g)
        e_list.append(e)
        ts_list.append(ts)

    g_stack = torch.stack(g_list, dim=0)  # (B,L,V,H,W)
    e_stack = torch.stack(e_list, dim=0)  # (B,L,V,H,W)

    B, L, V, H, W = g_stack.shape
    g_batch = g_stack.reshape(B * L, V, H, W)
    e_batch = e_stack.reshape(B * L, V, H, W)

    # 每个时间戳复制 L 次，对齐 B*L
    ts_out = []
    for ts in ts_list:
        ts_out.extend([ts] * L)

    return g_batch, e_batch, ts_out



if __name__ == "__main__":
    # 1. 初始化Reader
    gfs_reader = GFSReader(
        start_dt="2020-01-01 00:00:00",
        end_dt="2020-01-02 18:00:00"
    )
    era5_reader = ERA5Reader(
        start_dt="2020-01-01 00:00:00",
        end_dt="2020-01-02 18:00:00"
    )
    
    # 2. 选择要训练的变量（可根据需求调整）
    train_vars = [
        "Temperature",              # 13层
        "10 metre U wind component" # 1层
    ]
    
    # 3. 初始化数据集
    dataset = GFSERA5PairDataset(
        gfs_reader=gfs_reader,
        era5_reader=era5_reader,
        gfs_vars=train_vars,
        normalize=False,
        base_layers=13,
        pad_mode="repeat"  # 推荐用repeat（物理意义更合理）
    )
    
    # 4. 测试单样本
    gfs_tensor, era5_tensor, ts_str = dataset[0]
    print(f"\n=== 单样本维度 ===")
    print(f"时间戳：{ts_str}")
    print(f"单样本形状（层数×变量数×纬度×经度）：{gfs_tensor.shape}")
    print(f"  - 层数维度：{gfs_tensor.shape[0]}（统一为13层）")
    print(f"  - 变量数（Channels）：{gfs_tensor.shape[1]}（{len(train_vars)}个）")
    print(f"  - 空间维度：{gfs_tensor.shape[2]}×{gfs_tensor.shape[3]}")
    
    # 5. 初始化DataLoader（batch_size=2）
    batch_size = 2
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        # 强制指定collate_fn，避免使用默认函数
        collate_fn=lambda x: collate_fn(x, base_layers=13)
    )
    
    # 6. 测试批量数据
    # 替换原有测试循环
    for batch_idx, (gfs_batch, era5_batch, ts_batch) in enumerate(dataloader):
        print(f"[DataLoader] batch_idx={batch_idx}, gfs_batch={tuple(gfs_batch.shape)}, era5_batch={tuple(era5_batch.shape)}")
        print(f"[DataLoader] merged_batch={gfs_batch.shape[0]} (应该等于 batch_size*13)")
        break
    
    # 7. 关闭资源
    gfs_reader.close()
    print("\n✅ 所有测试完成！")