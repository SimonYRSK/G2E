import xarray as xr
import numpy as np
import os
import shutil
import warnings
from pathlib import Path
import pandas as pd

warnings.filterwarnings('ignore')

# ===================== 配置 =====================
ERA5_RAW = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/huangqiusheng/datasets/era5.rtm.02_25.6h.c109.new3/"
GFS_RAW  = ERA5_RAW   # 测试时用 ERA5 自己，确保替换前后数值不变

# 输出目录
OUTPUT_ROOT = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/localreplaced/with_trans_gfs"

# 要替换的通道（根据需要修改）
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

# 时间范围（字符串格式，与 slice 兼容）
TIME_SLICE = ["20240101", "20240101"]   # 测试用单日

# =================================================

def align_direction(ds_era5, ds_gfs):
    """只翻转 lat/lon 方向，不改变坐标数值"""
    era5_lat_inc = ds_era5.lat[1] > ds_era5.lat[0] if len(ds_era5.lat) > 1 else True
    gfs_lat_inc  = ds_gfs.lat[1]  > ds_gfs.lat[0]  if len(ds_gfs.lat) > 1 else True

    era5_lon_inc = ds_era5.lon[1] > ds_era5.lon[0] if len(ds_era5.lon) > 1 else True
    gfs_lon_inc  = ds_gfs.lon[1]  > ds_gfs.lon[0]  if len(ds_gfs.lon) > 1 else True

    if era5_lat_inc != gfs_lat_inc:
        print("  → 翻转 GFS lat 方向以匹配 ERA5")
        ds_gfs = ds_gfs.isel(lat=slice(None, None, -1))

    if era5_lon_inc != gfs_lon_inc:
        print("  → 翻转 GFS lon 方向以匹配 ERA5")
        ds_gfs = ds_gfs.isel(lon=slice(None, None, -1))

    return ds_gfs


def create_local_replaced(
    era5_path=ERA5_RAW,
    gfs_path=GFS_RAW,
    output_root=OUTPUT_ROOT,
    target_channels=TARGET_CHANNELS,
    time_slice=TIME_SLICE,
    chunks={'time': 1, 'lat': 721, 'lon': 1440}
):
    print(f"输出目录: {output_root}")
    os.makedirs(output_root, exist_ok=True)

    # 1. 打开 ERA5
    ds_era5 = xr.open_zarr(era5_path, chunks=chunks)

    # 筛选时间范围
    ds_era5 = ds_era5.sel(time=slice(*time_slice))

    if len(ds_era5.time) == 0:
        print("时间范围无数据")
        return

    print(f"处理时间范围：{ds_era5.time[0].values} → {ds_era5.time[-1].values}")
    print(f"总通道数：{len(ds_era5.channel)}")
    print(f"是否有 time 索引：{'time' in ds_era5.indexes}")

    # 2. 打开 GFS
    ds_gfs = xr.open_zarr(gfs_path, chunks=chunks)
    ds_gfs = ds_gfs.sortby("time")
    ds_gfs = ds_gfs.sel(time=slice(*time_slice))

    # 3. 对齐方向
    ds_gfs = align_direction(ds_era5, ds_gfs)

    # 4. 检查坐标接近度
    if not np.allclose(ds_era5.lat.values, ds_gfs.lat.values, atol=0.1):
        print("警告：lat 坐标差异较大 (>0.1度)，可能需要手动对齐")
    if not np.allclose(ds_era5.lon.values, ds_gfs.lon.values, atol=0.1):
        print("警告：lon 坐标差异较大 (>0.1度)，可能需要手动对齐")

    # 5. 批量替换所有目标通道（一次性操作）
    print("\n开始批量替换目标通道...")
    for ch in target_channels:
        if ch not in ds_era5.channel.values:
            print(f"  跳过：通道 {ch} 不存在于 ERA5")
            continue
        if ch not in ds_gfs.channel.values:
            print(f"  跳过：通道 {ch} 不存在于 GFS")
            continue

        try:
            # 提取 GFS 数据（所有时间步，已对齐方向）
            gfs_data = ds_gfs["data"].sel(channel=ch).compute()

            # 替换到 ERA5（沿 channel 赋值，自动广播到 time 维度）
            ds_era5["data"].loc[dict(channel=ch)] = gfs_data

            print(f"  已替换通道 {ch}（{len(ds_era5.time)} 个时间步）")
        except Exception as e:
            print(f"  替换 {ch} 失败：{e}")
            continue

    # 6. 一次性保存整个数据集
    output_path = os.path.join(output_root, "era5_localreplaced.zarr")
    print(f"\n一次性保存到：{output_path}")

    # 先删除旧的（避免脏数据干扰）
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        print("已清理旧的 zarr 目录")

    ds_era5.to_zarr(
        output_path,
        mode="w",
        consolidated=True,
    )

    # 7. 复制 mean.nc / std.nc（如果存在）
    for fname in ["mean.nc", "std.nc"]:
        src = os.path.join(era5_path, fname)
        dst = os.path.join(output_path, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"已复制 {fname}")

    print("\n替换完成！")
    print(f"输出路径：{output_path}")
    print(f"通道数：{len(ds_era5.channel)}")
    print(f"时间步数：{len(ds_era5.time)}")


if __name__ == "__main__":
    # 执行替换
    create_local_replaced(
        era5_path=ERA5_RAW,
        gfs_path="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/G2E/inferenced/baseline2_2",   #/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/G2E/inferenced/baseline1_25   /cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/database/gfs_2020_2024_c70_normalized     
        time_slice=["20240101", "20240101"]
    )

    # 测试对比
    print("\n=== 开始验证替换前后差异 ===")
    zarr_orig = xr.open_zarr(ERA5_RAW)
    output_path = os.path.join(OUTPUT_ROOT, "era5_localreplaced.zarr")
    zarr_new  = xr.open_zarr(output_path)

    # 使用 Timestamp 创建时间对象，更可靠
    t = pd.Timestamp("2024-01-01 12:00:00")
    ch = "t1000"

    try:
        orig = zarr_orig["data"].sel(time=t, channel=ch).compute()
        newv = zarr_new["data"].sel(time=t, channel=ch).compute()

        diff = np.abs(orig - newv)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())

        arr1 = orig.data
        arr2 = newv.data

        print(f"通道 {ch} @ {t}")
        print(f"最大差异：{max_diff:.10f}")
        print(f"平均差异：{mean_diff:.10f}")
        print(f"是否几乎相同（<1e-5）：{max_diff < 1e-5}")
        print(f"是否有 time 索引（新 zarr）：{'time' in zarr_new.indexes}")
        print(f"原始max={np.nanmax(arr1):.6f}, 替换后max={np.nanmax(arr2):.6f}, 差值max={np.nanmax(np.abs(arr1-arr2)):.6e}")
        print(f"原始min={np.nanmin(arr1):.6f}, 替换后min={np.nanmin(arr2):.6f}, 差值min={np.nanmin(np.abs(arr1-arr2)):.6e}")
    except Exception as e:
        print(f"验证失败：{e}")
        print("请检查 zarr 是否正确生成，或 time 值是否匹配")

    # 关闭数据集
    zarr_orig.close()
    zarr_new.close()

    print("\n=== 检查 lat/lon 方向对齐正确性 ===")

    # 1. 打印 ERA5 原生 lat 顺序（确认是 90 -> -90）
    print("ERA5 lat 顺序（前5 + 后5）：")
    print(zarr_orig.lat.values[:5])
    print(zarr_orig.lat.values[-5:])

    # 2. 取固定经度（例如 lon ≈ 0），沿纬度看温度分布
    fixed_lon = 0.0  # 或选择你关心的经度
    orig_profile = orig.sel(lon=fixed_lon, method='nearest')
    new_profile  = newv.sel(lon=fixed_lon, method='nearest')

    # 沿纬度排序打印（确保 lat 从高到低）
    lat_sorted_idx = np.argsort(orig_profile.lat.values)[::-1]  # 从北到南
    print(f"\n固定经度 ≈ {fixed_lon:.1f}°E 的纬向剖面（从北极到南极）：")
    print("纬度       ERA5 t50     替换后 t50     差异")
    for idx in lat_sorted_idx[:10]:  # 只打印前10个高纬示例
        lat_val = orig_profile.lat.values[idx]
        o_val = orig_profile.values[idx]
        n_val = new_profile.values[idx]
        print(f"{lat_val:6.1f}°   {o_val:8.3f}      {n_val:8.3f}     {n_val - o_val:8.3f}")

    # 3. 检查极值位置（北极 vs 南极）
    era5_north_pole = orig.isel(lat=0).values.mean()   # 假设 lat[0] 是北极
    era5_south_pole = orig.isel(lat=-1).values.mean()
    new_north_pole  = newv.isel(lat=0).values.mean()
    new_south_pole  = newv.isel(lat=-1).values.mean()

    print(f"\n北极附近平均 t50 (lat≈90°):  ERA5 = {era5_north_pole:.3f} K, 替换后 = {new_north_pole:.3f} K")
    print(f"南极附近平均 t50 (lat≈-90°): ERA5 = {era5_south_pole:.3f} K, 替换后 = {new_south_pole:.3f} K")

    # 4. 如果北极和南极的相对关系颠倒了，说明翻转失败
    if (new_north_pole > new_south_pole and era5_north_pole < era5_south_pole) or \
        (new_north_pole < new_south_pole and era5_north_pole > era5_south_pole):
        print("⚠️ 警告：极值分布可能颠倒！lat 方向对齐可能失败")
    else:
        print("✓ 极值分布方向合理（北极与南极相对关系保持一致）")