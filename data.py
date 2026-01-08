import zarr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 先复用你定义的映射表（保持不变）
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
                start_dt: str = "2020-01-01 00:00:00",  # 修正类型标注
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
            
            # ========== 关键修复：兼容字符串/字节串 ==========
            # 读取channel名，自动判断是字符串还是字节串
            channel_raw = self.channel_zarr[:]
            self.channel_names = []
            for name in channel_raw:
                if isinstance(name, bytes):
                    # 字节串 → 解码为字符串
                    self.channel_names.append(name.decode('utf-8'))
                elif isinstance(name, (str, np.str_)):
                    # 直接是字符串 → 无需解码
                    self.channel_names.append(str(name))
                else:
                    # 其他类型 → 强制转字符串
                    self.channel_names.append(str(name))
            
            # 构建channel名→索引映射（关键：通过channel名找数据索引）
            self.channel_name2idx = {name: idx for idx, name in enumerate(self.channel_names)}
            
            # 筛选有效时间索引
            self.valid_time_indices = self._time_filter()
            print(f"✅ ERA5加载完成：")
            print(f"   有效时间步：{len(self.valid_time_indices)}个（{self.start_dt}~{self.end_dt}）")
            print(f"   总通道数：{self.n_channels}，空间维度：{self.lat_size}×{self.lon_size}")
            print(f"   前5个通道名：{self.channel_names[:5]}")  # 新增：打印通道名，方便校验
        
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
            # 索引→时间：基准时间 + 索引×时间步
            delta_hours = time_idx * self.time_step_hours
            timestamp = self.base_time + timedelta(hours=delta_hours)
            valid_timestamps.append(timestamp)
        return valid_timestamps

    def _get_nearest_time_idx(self, target_time: datetime) -> int:
        """根据目标时间找最近的有效时间索引"""
        # 计算目标时间对应的理论索引
        delta_hours = (target_time - self.base_time).total_seconds() / 3600
        target_idx = int(delta_hours // self.time_step_hours)
        
        # 找有效索引中最接近的（避免索引越界）
        valid_indices_arr = np.array(self.valid_time_indices)
        nearest_idx = valid_indices_arr[np.argmin(np.abs(valid_indices_arr - target_idx))]
        
        # 校验是否在有效范围内
        if nearest_idx not in self.valid_time_indices:
            raise ValueError(f"目标时间{target_time}无对应的有效ERA5数据")
        return nearest_idx

    def read_by_time(self, target_time: datetime, gfs_vars: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        按目标时间读取ERA5数据，返回{GFS通用变量名: (层数, lat, lon)}
        Args:
            target_time: 目标时间（datetime对象）
            gfs_vars: 要读取的GFS通用变量名列表（默认读取所有）
        Returns:
            字典：{GFS通用变量名: 数组(层数, lat, lon)}
        """
        gfs_vars = gfs_vars or list(self.gfs2era5_vars.keys())
        
        # 1. 找到最近的有效时间索引
        time_idx = self._get_nearest_time_idx(target_time)
        
        # 2. 逐个读取GFS通用变量对应的ERA5分层变量
        result = {}
        for gfs_var in gfs_vars:
            if gfs_var not in self.gfs2era5_vars:
                raise ValueError(f"ERA5不支持GFS通用变量：{gfs_var}")
            
            # 获取该GFS变量对应的ERA5分层变量列表
            era5_vars = self.gfs2era5_vars[gfs_var]
            layer_data_list = []
            
            for era5_var in era5_vars:
                # 找ERA5变量对应的channel索引
                if era5_var not in self.channel_name2idx:
                    raise ValueError(f"ERA5无该通道：{era5_var}（所有通道：{list(self.channel_name2idx.keys())[:10]}...）")
                chan_idx = self.channel_name2idx[era5_var]
                
                # 读取数据：(time, channel, lat, lon) → 取指定time和channel
                # 懒加载：只读取需要的切片，不加载全量数据
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

# ===================== 测试代码 =====================
if __name__ == '__main__':
    # 初始化ERA5Reader
    era5_reader = ERA5Reader(
        start_dt="2020-01-01 00:00:00",
        end_dt="2024-01-02 18:00:00"
    )
    
    # 测试读取指定时间的数据
    target_time = datetime(2020, 1, 1, 6, 0, 0)
    data = era5_reader.read_by_time(
        target_time=target_time,
        gfs_vars=["Temperature", "2 metre temperature"]
    )
    
    # 打印结果
    print(f"\n=== 读取结果 ===")
    for gfs_var, arr in data.items():
        print(f"{gfs_var} 形状：{arr.shape}")
    print(f"有效时间戳示例：{era5_reader.time_index[:5]}")