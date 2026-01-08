import xarray as xr
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

class GFSReader:
    def __init__(self, zarr_path: str, mapping: Dict = gfs2era5_mapping):
        self.zarr_path = zarr_path
        self.mapping = mapping  # 传入gfs2era5映射表
        self.ds = None  # 延迟加载的Dataset
        self.time_index = None  # 统一时间索引
        self.supported_gfs_vars = list(mapping.keys())  # 支持的GFS变量名

    def _load_data(self):
        """延迟加载GFS数据，解析时间"""
        if self.ds is None:
            self.ds = xr.open_zarr(self.zarr_path, consolidated=True)
            self.time_index = pd.to_datetime(self.ds['time'].values).sort_values()
            print(f"✅ GFS数据加载完成")
            print(f"   时间范围：{self.time_index.min()} ~ {self.time_index.max()}")
            print(f"   支持的变量：{self.supported_gfs_vars}")

    def read_by_time(self, target_time: pd.Timestamp, gfs_vars: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        按时间读取GFS变量，返回{GFS通用变量名: (层数, lat, lon)}
        Args:
            target_time: 目标时间（统一格式）
            gfs_vars: 要读取的GFS变量名列表（默认读取所有支持的变量）
        """
        self._load_data()
        gfs_vars = gfs_vars or self.supported_gfs_vars
        
        # 1. 找到GFS中最接近的时间
        nearest_idx = np.argmin(np.abs(self.time_index - target_time))
        nearest_time = self.time_index[nearest_idx]

        # 2. 读取指定变量
        result = {}
        for gfs_var in gfs_vars:
            if gfs_var not in self.mapping:
                raise ValueError(f"GFS不支持变量{gfs_var}，支持的变量：{self.supported_gfs_vars}")
            if gfs_var not in self.ds:
                raise ValueError(f"GFS Zarr中没有变量{gfs_var}")
            
            # 读取变量数据
            var_data = self.ds[gfs_var].sel(time=nearest_time).values.squeeze()
            
            # 处理维度：surface变量（无层数）→ 扩展为(1, lat, lon)，upper_air变量保持(13, lat, lon)
            var_type = self.mapping[gfs_var]["var_type"]
            if var_type == "surface" and var_data.ndim == 2:
                var_data = np.expand_dims(var_data, axis=0)  # (1, lat, lon)
            elif var_type == "upper_air" and var_data.ndim == 2:
                raise ValueError(f"GFS变量{gfs_var}是高空变量，但数据无层数维度！")
            
            result[gfs_var] = var_data
            print(f"📌 GFS变量{gfs_var}形状：{var_data.shape}（{self.mapping[gfs_var]['var_type']}）")
        
        return result

    @property
    def all_gfs_vars(self) -> List[str]:
        """返回所有支持的GFS变量名"""
        return self.supported_gfs_vars


import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

class GFSERA5PairDataset(Dataset):
    def __init__(
        self,
        gfs_reader: GFSReader,
        era5_reader: ERA5Reader,
        gfs_vars: Optional[List[str]] = None,
        time_window: Tuple[str, str] = ("2020-01-01", "2020-01-31"),
        time_diff_threshold: int = 3600,  # 时间差阈值（秒）
        normalize: bool = True,
        spatial_shape: Tuple[int, int] = None  # 统一空间分辨率（lat, lon）
    ):
        self.gfs_reader = gfs_reader
        self.era5_reader = era5_reader
        self.gfs_vars = gfs_vars or gfs_reader.all_gfs_vars
        self.time_diff_threshold = time_diff_threshold
        self.normalize = normalize
        self.spatial_shape = spatial_shape  # 统一空间分辨率（可选）

        # 1. 加载时间索引并筛选时间窗口
        self.gfs_reader._load_data()
        self.era5_reader._load_data()
        start = pd.to_datetime(time_window[0])
        end = pd.to_datetime(time_window[1])
        self.gfs_times = gfs_reader.time_index[(gfs_reader.time_index >= start) & (gfs_reader.time_index <= end)]
        self.era5_times = era5_reader.time_index[(era5_reader.time_index >= start) & (era5_reader.time_index <= end)]

        # 2. 生成时间配对（按时间差阈值）
        self.paired_times = self._generate_paired_times()
        if not self.paired_times:
            raise ValueError("❌ 没有找到符合时间差阈值的配对样本")
        print(f"✅ 共找到{len(self.paired_times)}个时间配对样本")

        # 3. 预计算标准化参数（基于ERA5）
        self.norm_params = self._compute_norm_params() if normalize else None

        # 4. 预计算总层数（所有变量的层数之和，用于batch维度拼接）
        self.total_layers = self._compute_total_layers()
        print(f"✅ 所有变量总层数：{self.total_layers}")

    def _generate_paired_times(self) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """生成时间配对（GFS时间, ERA5时间）"""
        paired_times = []
        for gfs_time in self.gfs_times:
            time_diffs = np.abs((self.era5_times - gfs_time).total_seconds())
            min_diff = np.min(time_diffs)
            if min_diff <= self.time_diff_threshold:
                era5_time = self.era5_times[np.argmin(time_diffs)]
                paired_times.append((gfs_time, era5_time))
        return paired_times

    def _compute_total_layers(self) -> int:
        """计算所有变量的层数之和（如：Temperature(13) + t2m(1) + ...）"""
        total = 0
        for gfs_var in self.gfs_vars:
            # 从映射表获取层数（ERA5变量列表长度 = 层数）
            total += len(self.gfs_reader.mapping[gfs_var]["era5_vars"])
        return total

    def _compute_norm_params(self) -> Dict[str, Tuple[float, float]]:
        """按GFS通用变量计算标准化参数（均值+标准差）"""
        norm_params = {}
        # 取前10个样本计算统计量
        sample_times = self.paired_times[:10]
        for gfs_var in self.gfs_vars:
            all_data = []
            for _, era5_time in sample_times:
                data = self.era5_reader.read_by_time(era5_time, [gfs_var])[gfs_var]
                all_data.append(data)
            all_data = np.concatenate(all_data, axis=0)
            norm_params[gfs_var] = (np.mean(all_data), np.std(all_data))
        return norm_params

    def _normalize(self, data: np.ndarray, gfs_var: str) -> np.ndarray:
        """标准化单变量数据"""
        mean, std = self.norm_params[gfs_var]
        return (data - mean) / (std + 1e-8)  # 避免除0

    def _resize_spatial(self, tensor: torch.Tensor) -> torch.Tensor:
        """统一空间分辨率（可选）"""
        if self.spatial_shape is None:
            return tensor
        # tensor形状：(层数, lat, lon) → 转成(1, 层数, lat, lon)做插值 → 恢复(层数, lat, lon)
        tensor = tensor.unsqueeze(0)
        tensor = F.interpolate(tensor, size=self.spatial_shape, mode="bilinear", align_corners=False)
        return tensor.squeeze(0)

    def __len__(self) -> int:
        return len(self.paired_times)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 1. 获取配对时间
        gfs_time, era5_time = self.paired_times[idx]
        
        # 2. 读取GFS和ERA5数据（{GFS通用变量: (层数, lat, lon)}）
        gfs_data = self.gfs_reader.read_by_time(gfs_time, self.gfs_vars)
        era5_data = self.era5_reader.read_by_time(era5_time, self.gfs_vars)
        
        # 3. 标准化 + 空间分辨率统一
        gfs_tensors = []
        era5_tensors = []
        for gfs_var in self.gfs_vars:
            # 标准化
            if self.normalize:
                gfs_var_data = self._normalize(gfs_data[gfs_var], gfs_var)
                era5_var_data = self._normalize(era5_data[gfs_var], gfs_var)
            else:
                gfs_var_data = gfs_data[gfs_var]
                era5_var_data = era5_data[gfs_var]
            
            # 转tensor并统一空间分辨率
            gfs_tensor = torch.tensor(gfs_var_data, dtype=torch.float32)
            era5_tensor = torch.tensor(era5_var_data, dtype=torch.float32)
            gfs_tensor = self._resize_spatial(gfs_tensor)
            era5_tensor = self._resize_spatial(era5_tensor)
            
            # 添加到列表（后续拼接所有变量的层数）
            gfs_tensors.append(gfs_tensor)
            era5_tensors.append(era5_tensor)
        
        # 4. 拼接所有变量的层数 → (总层数, lat, lon)
        gfs_combined = torch.cat(gfs_tensors, axis=0)  # (总层数, lat, lon)
        era5_combined = torch.cat(era5_tensors, axis=0)  # (总层数, lat, lon)
        
        # 5. 把“总层数”维度压到batch维度（核心！）
        # 方式1：返回(总层数, lat, lon)，后续DataLoader会自动拼batch → (batch_size×总层数, lat, lon)
        # 方式2：显式扩展batch维度 → (总层数, 1, lat, lon)，后续拼接为(batch_size, 总层数, lat, lon)
        # 这里用方式1（更贴合你的“压到batch维度”需求）
        
        return {
            "gfs": gfs_combined,                # (总层数, lat, lon)
            "era5": era5_combined,              # (总层数, lat, lon)
            "gfs_time": gfs_time.strftime("%Y-%m-%d %H:%M:%S"),
            "era5_time": era5_time.strftime("%Y-%m-%d %H:%M:%S"),
            "gfs_vars": self.gfs_vars,
            "total_layers": self.total_layers
        }