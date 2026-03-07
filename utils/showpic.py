import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

ERA5_PATH = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/huangqiusheng/datasets/era5.rtm.02_25.6h.c109.new3/"
GFS_PATH  = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/database/gfs_2020_2024_c70_normalized"
TARGET_CHANNELS = [
    "t50", "t100", "t200", "t500", "t850",
    "u200", "v500", "q700", "z500"
]
# 选一个时间点
TIME = "2024-01-01T12:00:00"

# 打开数据
ds_era5 = xr.open_zarr(ERA5_PATH)
ds_gfs  = xr.open_zarr(GFS_PATH)

save_dir = "./compare_era5_gfs"
os.makedirs(save_dir, exist_ok=True)

for ch in TARGET_CHANNELS:
    if ch not in ds_era5.channel.values or ch not in ds_gfs.channel.values:
        print(f"跳过 {ch}，该通道不存在于ERA5或GFS")
        continue

    arr_era5 = ds_era5["data"].sel(time=TIME, channel=ch)
    arr_gfs  = ds_gfs["data"].sel(time=TIME, channel=ch)

    # 可视化
    plt.figure(figsize=(18,5))
    plt.subplot(1,3,1)
    plt.title(f"ERA5 {ch}")
    plt.imshow(arr_era5, origin='lower', cmap='jet')
    plt.colorbar()

    plt.subplot(1,3,2)
    plt.title(f"GFS {ch}")
    plt.imshow(arr_gfs, origin='lower', cmap='jet')
    plt.colorbar()

    diff = arr_era5 - arr_gfs
    plt.subplot(1,3,3)
    plt.title(f"Diff (ERA5-GFS) {ch}")
    plt.imshow(diff, origin='lower', cmap='bwr')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{ch}_compare.png"))
    plt.close()

    # 差异直方图
    plt.figure(figsize=(6,4))
    plt.hist(diff.values.flatten(), bins=100, color='gray', alpha=0.8)
    plt.title(f"Diff Histogram (ERA5-GFS) {ch}")
    plt.xlabel("Difference")
    plt.ylabel("Count")
    plt.grid()
    plt.savefig(os.path.join(save_dir, f"{ch}_diff_hist.png"))
    plt.close()

    print(f"{ch} 可视化与直方图已保存。")

print("全部完成！")