import zarr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import xarray as xr  # GFS用xarray读取更方便

# 复用映射表（和ERA5Reader一致）
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

# 反向映射（备用）
era52gfs_mapping = {}
for gfs_var, info in gfs2era5_mapping.items():
    for era5_var in info["era5_vars"]:
        era52gfs_mapping[era5_var] = gfs_var

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
        self.valid_time_stamps = None   # 筛选后的时间戳列表（和valid_time_indices一一对应）
        self.level_order = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]  # 固定层级顺序
        
        # 初始化加载
        self._load_zarr_handles()
        self._filter_valid_times()

    def _load_zarr_handles(self):
        """懒加载GFS Zarr数据（xarray方式）"""
        try:
            # 核心修复：移除mode='r'（xarray的open_zarr不支持该参数）
            self.ds = xr.open_zarr(self.gfs_root)
            print(f"✅ GFS Zarr加载完成：路径={self.gfs_root}")
            
            # 验证核心维度
            required_dims = ["time", "lat", "lon", "level"]
            missing_dims = [d for d in required_dims if d not in self.ds.dims]
            if missing_dims:
                raise ValueError(f"GFS Zarr缺失核心维度：{missing_dims}")
            
            # 构建完整的时间索引（原始数据的所有时间）
            self.full_time_index = pd.DatetimeIndex(self.ds["time"].values.astype('datetime64[s]'))
            print(f"   GFS总时间范围：{self.full_time_index[0]} ~ {self.full_time_index[-1]}")
            print(f"   GFS总时间步：{len(self.full_time_index)}")
            print(f"   GFS空间维度：lat={self.ds.dims['lat']}, lon={self.ds.dims['lon']}")
        
        except Exception as e:
            raise RuntimeError(f"加载GFS Zarr失败：{str(e)}")

    def _filter_valid_times(self):
        """筛选指定时间范围内的有效时间索引和时间戳"""
        # 找到start_dt和end_dt之间的时间索引（原始数据的索引）
        mask = (self.full_time_index >= self.start_dt) & (self.full_time_index <= self.end_dt)
        self.valid_time_indices = np.where(mask)[0].tolist()
        # 对应的时间戳列表（和valid_time_indices一一对应）
        self.valid_time_stamps = self.full_time_index[mask].tolist()
        
        if not self.valid_time_indices:
            raise ValueError(f"GFS无有效时间数据：{self.start_dt}~{self.end_dt}")
        
        # 对外暴露的time_index（和ERA5Reader对齐）
        self.time_index = pd.DatetimeIndex(self.valid_time_stamps)
        print(f"✅ GFS有效时间步：{len(self.valid_time_indices)}个（{self.start_dt}~{self.end_dt}）")

    def _get_nearest_time_idx(self, target_time: datetime) -> Tuple[int, datetime]:
        """
        找到目标时间最近的有效时间索引和对应的时间戳
        Returns:
            (原始数据索引, 对应的时间戳)
        """
        # 计算目标时间与所有有效时间戳的差值（秒）
        time_diffs = [abs((ts - target_time).total_seconds()) for ts in self.valid_time_stamps]
        # 找到最小差值的索引
        nearest_pos = np.argmin(time_diffs)
        # 原始数据索引
        nearest_raw_idx = self.valid_time_indices[nearest_pos]
        # 对应的时间戳
        nearest_ts = self.valid_time_stamps[nearest_pos]
        return nearest_raw_idx, nearest_ts

    def read_by_time(self, target_time: datetime, gfs_vars: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        按目标时间读取GFS数据（和ERA5Reader接口完全对齐）
        Args:
            target_time: 目标时间（datetime对象）
            gfs_vars: 要读取的GFS变量列表（默认所有）
        Returns:
            字典：{GFS变量名: 数组(层数, lat, lon)}
        """
        # 处理默认变量列表
        gfs_vars = gfs_vars or self.all_gfs_vars_list
        
        # 1. 找到最近的有效时间索引和时间戳（核心修复：避免索引查找失败）
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
                var_data_sorted = var_data.isel(level=level_indices)  # 按层级顺序排序
                var_arr = var_data_sorted.values  # (13, 721, 1440)
            else:
                # 地面变量：增加层数维度 → (1, 721, 1440)（和ERA5对齐）
                var_arr = var_data.values[np.newaxis, :, :]  # 扩展维度
            
            # 处理缺失值（填充为0，可根据需求调整）
            var_arr = np.nan_to_num(var_arr, nan=0.0)
            result[gfs_var] = var_arr
            print(f"   GFS变量{gfs_var}：形状{var_arr.shape}（{var_type}）")
        
        return result

    @property
    def all_gfs_vars(self) -> List[str]:
        """返回所有支持的GFS变量名（和ERA5Reader对齐）"""
        return self.all_gfs_vars_list

    def close(self):
        """关闭Dataset句柄，释放资源"""
        if self.ds is not None:
            self.ds.close()
            print("✅ GFS Dataset已关闭")

# ===================== 测试代码 =====================
if __name__ == '__main__':
    # 初始化GFSReader
    gfs_reader = GFSReader(
        start_dt="2020-01-01 00:00:00",
        end_dt="2020-01-02 18:00:00"
    )
    
    # 测试读取指定时间的数据
    target_time = datetime(2020, 1, 1, 6, 0, 0)
    data = gfs_reader.read_by_time(
        target_time=target_time,
        gfs_vars=["Temperature", "2 metre temperature", "Geopotential height"]
    )
    
    # 打印结果
    print(f"\n=== GFS读取结果 ===")
    for gfs_var, arr in data.items():
        print(f"{gfs_var} 形状：{arr.shape}")
    
    # 打印时间索引示例
    print(f"\n有效时间戳示例：{gfs_reader.time_index[:5]}")
    
    # 关闭句柄
    gfs_reader.close()