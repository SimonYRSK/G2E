import os
import torch
import xarray as xr
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
import sys
import argparse
from collections import OrderedDict




def chunk_time(ds):
    dims = {k: v for k, v in ds.dims.items()}
    dims['time'] = 1
    ds = ds.chunk(dims)
    return ds


def get_date_str():
    dates_str = []
    start_time = datetime.datetime.strptime(args.years[0], "%Y%m%d%H")
    end_time = datetime.datetime.strptime(args.years[1], "%Y%m%d%H")
    for date in pd.date_range(start_time, end_time, freq=f"{args.step}h"):
        date = date.strftime("%Y%m%d%H")
        dates_str.append(date)
    return dates_str


def time_encoding(init_time, total_step, freq=6):
    init_time = np.array([init_time])
    tembs = []
    for i in range(total_step):
        hours = np.array([pd.Timedelta(hours=t * freq) for t in [i - 1, i, i + 1]])
        times = init_time[:, None] + hours[None]
        times = [pd.Period(t, 'h') for t in times.reshape(-1)]
        times = [(p.day_of_year / 366, p.hour / 24) for p in times]
        temb = np.array(times, dtype=np.float32)
        temb = np.concatenate([np.sin(temb), np.cos(temb)], axis=-1)
        temb = temb.reshape(1, -1)
        tembs.append(temb)
    return np.stack(tembs)


def define_model_forecast():
    sys.path.append(f"{args.model_forecast_dir}")
    from fuxi.fuxi_grad import UTransformer, FuXi
    print("define model forecast")
    # model param
    in_chans = 75
    out_chans = 70
    in_frames = 2
    image_size = (720, 1440)
    window_size = 9
    patch_size = 4
    down_times = 1
    embed_dim = 1536
    num_heads = 24
    depths = [12, 12, 12, 12]
    out_frames = 1
    step_range = args.step_range
    # data param
    conds = np.load(f"fuxi/conds.npy")
    std = np.load(f"fuxi/std.npy")
    mean = np.load(f"fuxi/mean.npy")
    const = torch.from_numpy(conds).to(dtype=dtype_forecast, device=device)
    std = torch.from_numpy(std).to(device)
    mean = torch.from_numpy(mean).to(device)

    decoder = UTransformer(
        in_chans=in_chans,
        out_chans=out_chans,
        in_frames=in_frames,
        image_size=image_size,
        window_size=window_size,
        patch_size=patch_size,
        down_times=down_times,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depths=depths,
    )
    model = FuXi(
        in_frames=in_frames,
        out_frames=out_frames,
        step_range=step_range,
        decoder=[decoder, decoder, decoder],
        const=const,
        std=std,
        mean=mean,
        device=device,
        dtype=dtype_forecast,
    ).to(dtype=dtype_forecast, device=device).eval()

    model.load(args.model_forecast_dir, fmt='pth')
    return model

# def reverse_standardization_np(data):
#     mean = np.load(f"/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/xuxiaoze/data_prep/era5/mean_era5.npy")
#     std = np.load(f"/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/xuxiaoze/data_prep/era5/std_era5.npy")
#     data = (data * std + mean)
#     data[-1] = np.exp(data[-1]) - 1
#     return data


# def read_input_era5(date):
#     ds_era5 = xr.open_zarr(args.era5_dir)
#     ds_era5 = ds_era5.sel(time=[date - (datetime.timedelta(hours=6)), date])
#     input = ds_era5.z.values
#     input = torch.from_numpy(input)
#     return input
def safe_open_mean_std(base_path, mean_name="mean.nc", std_name="std.nc", var_list=None):
    """读取 mean/std 并筛选 channel"""
    try:
        mean_da = xr.open_dataset(os.path.join(base_path, mean_name))['mean']
        std_da  = xr.open_dataset(os.path.join(base_path, std_name))['std']
        # mean_da = update_dims(mean_da)
        # std_da  = update_dims(std_da)
        if var_list is not None:
            mean_da = mean_da.sel(channel=var_list)
            std_da  = std_da.sel(channel=var_list)
        else:
            raise ValueError("mean/std 数据中缺少 'channel' 或 'level' 坐标")
        return mean_da.values.astype(np.float32), std_da.values.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"加载 mean/std 失败: {base_path}, {e}")

def normalize(args, x, inv=False, channel=None):
    mean, std = safe_open_mean_std(args.era5_dir, "mean.nc", "std.nc", var_list=channel)
    mean = mean[:, None, None]
    std = std[:, None, None]
    
    if inv:
        x = x * std + mean  
        tp = x[:, -1].clamp(min=0, max=7)
        x[:, -1] = tp.exp() - 1
    else:
        tp = x[:, -1]
        tp = torch.log(1 + tp.clamp(min=0))
        x[:, -1] = tp
        x = (x - mean) / std

    return x

def read_input_era5(args, date, channel):
    ds_era5 = xr.open_zarr(args.era5_dir, consolidated=False)
    time_sel = [date - datetime.timedelta(hours=6), date]
    if 'channel' in ds_era5.dims or 'channel' in ds_era5.coords:
        ds_era5 = ds_era5.sel(time=time_sel, channel=channel)
    elif 'level' in ds_era5.dims or 'level' in ds_era5.coords:
        ds_era5 = ds_era5.sel(time=time_sel, level=channel)
    else:
        raise KeyError(f"Dataset 缺少 channel/level 维度，当前维度: {list(ds_era5.dims)}")

    if 'channel' in ds_era5.coords:
        print(f"Selected Channel values for {date}: {ds_era5.channel.values}")
    check_output1(ds_era5)

    if 'data' in ds_era5.data_vars:
        input_data = torch.from_numpy(ds_era5['data'].values)
    elif 'z' in ds_era5.data_vars:
        input_data = torch.from_numpy(ds_era5['z'].values)
    else:
        first_var = list(ds_era5.data_vars)[0]
        input_data = torch.from_numpy(ds_era5[first_var].values)

    input_denorm = normalize(args, input_data, inv=True, channel=channel)
    
    return input_denorm


def _get_primary_data_var(ds: xr.Dataset) -> str:
    if 'z' in ds.data_vars:
        return 'z'
    if 'data' in ds.data_vars:
        return 'data'
    if len(ds.data_vars) > 0:
        return list(ds.data_vars)[0]
    raise ValueError("Dataset 不包含任何 data_vars")


def check_output(data):
    check_names = [
        'Z500', 'Z850',
        'T500', 'T850',
        'U500', 'U850',
        'V500', 'V850',
        'R500', 'R850',
        'T2M', 'U10', 'V10', 'MSL', 'TP']
    var_name = _get_primary_data_var(data)
    da = data[var_name]
    for i,lvl in enumerate(data.channel.values):
        if lvl.upper() in check_names:
            sel_kwargs = {'channel': lvl}
            if 'step' in da.dims:
                sel_kwargs['step'] = min(19, da.sizes['step'] - 1)
            v = da.sel(**sel_kwargs).values
            print(f'\033[92m{lvl}-{i}: {v.shape}, {v.min():.3f} ~ {v.max():.3f}\033[0m')

def check_output1(data):
    check_names = [
        'Z500', 'Z850',
        'T500', 'T850',
        'U500', 'U850',
        'V500', 'V850',
        'R500', 'R850',
        'T2M', 'U10', 'V10', 'MSL', 'TP']
    var_name = _get_primary_data_var(data)
    da = data[var_name]

    if 'level' in data.dims:
        for i,lvl in enumerate(data.level.values):
            sel_kwargs = {'level': lvl}
            if 'step' in da.dims:
                sel_kwargs['step'] = min(19, da.sizes['step'] - 1)
            v = da.sel(**sel_kwargs).values
            print(f'{lvl}-{i}: {v.shape}, {v.min():.3f} ~ {v.max():.3f}')
    else:
        for i,lvl in enumerate(data.channel.values):
            sel_kwargs = {'channel': lvl}
            if 'step' in da.dims:
                sel_kwargs['step'] = min(19, da.sizes['step'] - 1)
            v = da.sel(**sel_kwargs).values
            print(f'{lvl}-{i}: {v.shape}, {v.min():.3f} ~ {v.max():.3f}')

def forecast(date, input_data, model):
    # input_data = model.normalize(input_data)
    input_tmbs = torch.from_numpy(time_encoding(date, sum(args.step_range))).to(device=device, dtype=dtype_forecast)
    input = (input_data, input_tmbs)
    outputs = model.forward(input)
    print("outputs shape666:",outputs.shape)

   
    return outputs


def save_nc(ds, save_name, dtype=np.float32):
    from dask.diagnostics import ProgressBar
    ds = chunk_time(ds)
    ds = ds.astype(dtype)
    delayed_ds = ds.to_netcdf(save_name, compute=False)
    with ProgressBar():
        delayed_ds.compute()


# def save_forecast(date, output,channel):
#     output = output.cpu().detach().numpy()
#     lat=np.linspace(90, -90, 721)
#     lon=np.linspace(0, 359.75, 1440)
#     # lat=np.arange(90,-90,-0.25)
#     # lon=np.arange(0,360,0.25)
#     # lat = np.load('/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/xuxiaoze/data_prep/era5/latitude_era5.npy')[:, 0]
#     # lon = np.load('/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/xuxiaoze/data_prep/era5/longitude_era5.npy')[0, :]
#     ds_ana = xr.Dataset({'z': (['time', 'channel', 'lat', 'lon'], output.astype(np.float32))},
#                         coords={'time': [date], 'channel': channel, 'lat': lat, 'lon': lon})
#     ds_ana['time'].encoding['dtype'] = 'float64'
#     date_str = date.strftime("%Y%m%d%H")
#     ds_ana = chunk_time(ds_ana)
#     os.makedirs(f"{args.inference_dir}", exist_ok=True)
#     save_nc(ds_ana, f"{args.inference_dir}/{date_str}.nc", dtype=np.float32)
#     return None

def save_forecast(date, output_data,channel,num_steps):
    output_data = output_data.cpu().detach().numpy()
    lat=np.linspace(90, -90, 721)
    lon=np.linspace(0, 359.75, 1440)
    # lat=np.arange(90,-90,-0.25)
    # lon=np.arange(0,360,0.25)
    # lat = np.load('/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/xuxiaoze/data_prep/era5/latitude_era5.npy')[:, 0]
    # lon = np.load('/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/xuxiaoze/data_prep/era5/longitude_era5.npy')[0, :]
    print("output_data shape88:", output_data.shape)
    ds_ana = xr.Dataset(
        {'data': (['time', 'step', 'channel', 'lat', 'lon'], output_data.astype(np.float32))},
        coords={
            'time': [date],  # 根据时间步生成时间坐标
            'step': np.arange(num_steps),  # 这里将 step 作为一个维度（0, 1, 2, ..., num_steps-1）
            'channel': channel,
            'lat': lat,
            'lon': lon
        }
    )
    ds_ana['time'].encoding['dtype'] = 'float64'
    date_str = date.strftime("%Y%m%d%H")

    ds_ana = chunk_time(ds_ana)
    check_output(ds_ana)
    check_output1(ds_ana)
    if not os.path.exists(f"{args.inference_dir}"):
        os.makedirs(f"{args.inference_dir}", exist_ok=True)

    save_nc(ds_ana, f"{args.inference_dir}/{date_str}.nc", dtype=np.float32)
    return None



def main(args):
    dates_str = get_date_str()
    model_forecast = define_model_forecast()
    for date_str in tqdm(dates_str):
        date = pd.to_datetime(date_str, format="%Y%m%d%H")
        # try:
        input_forecast = read_input_era5(args,date=date,channel=args.channel)
        input_forecast = input_forecast.to(dtype=dtype_forecast, device=device)

        output_forecast = forecast(date=date, input_data=input_forecast, model=model_forecast)
        # output_forecast=output_forecast.squeeze()
        date_forecast = date
        # date_forecast = date
        num_steps=np.sum(args.step_range)
        save_forecast(date_forecast, output_forecast,args.channel,num_steps)

        print(f"[MaxMemory]: {torch.cuda.max_memory_allocated(device) / 1024 ** 3}")
        # except:
        #     print(f"{date_str} is wrong")



if __name__ == '__main__':
    import torch

    # 清空显存
    torch.cuda.empty_cache()

    channel = ['z50', 'z100', 'z150', 'z200', 'z250', 'z300', 'z400', 'z500',
               'z600', 'z700', 'z850', 'z925', 'z1000', 't50', 't100', 't150',
               't200', 't250', 't300', 't400', 't500', 't600', 't700', 't850',
               't925', 't1000', 'u50', 'u100', 'u150', 'u200', 'u250', 'u300',
               'u400', 'u500', 'u600', 'u700', 'u850', 'u925', 'u1000', 'v50',
               'v100', 'v150', 'v200', 'v250', 'v300', 'v400', 'v500', 'v600',
               'v700', 'v850', 'v925', 'v1000', 'r50', 'r100', 'r150', 'r200',
               'r250', 'r300', 'r400', 'r500', 'r600', 'r700', 'r850', 'r925',
               'r1000', 't2m', 'u10m', 'v10m', 'msl', 'tp']
    parser = argparse.ArgumentParser()
    parser.add_argument('--era5_dir', type=str,
                        default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/fanjiang/dataset/era5.2010_2025.c226.zarr")
    parser.add_argument('--inference_dir', type=str,
                        default='/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/G2E/inferenced/fuxi_pth_c70_inference')
    parser.add_argument('--model_forecast_dir', type=str,
                        default='/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/G2E/fuxi_inference/main/model/')
    parser.add_argument('--years', type=str, nargs="+", default=['2023050500', '2023050500'])
    parser.add_argument('--step', type=int, default=6)
    parser.add_argument('--step_range', type=int, nargs="+", default=[1])
    parser.add_argument('--channel', type=str, nargs="+", default=channel)
    args = parser.parse_args()
    device = 'cuda'
    dtype_forecast = torch.float32
    main(args)
    torch.cuda.empty_cache()
