import argparse
import os
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

LINE_RE = re.compile(
    r"Step\s+(?P<step>\d+):"
    r"Naive ERA5 RMSE=(?P<rmse_era5>[0-9.]+), ACC=(?P<acc_era5>[0-9.]+) \| "
    r"Naive GFS RMSE=(?P<rmse_naive>[0-9.]+), ACC=(?P<acc_naive>[0-9.]+) \| "
    r"GFS2ERA5 RMSE=(?P<rmse_trans>[0-9.]+), ACC=(?P<acc_trans>[0-9.]+)"
)

def parse_metrics_txt(txt_path):
    steps = []
    rmse_era5 = []
    rmse_naive = []
    rmse_trans = []
    acc_era5 = []
    acc_naive = []
    acc_trans = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            steps.append(int(m.group("step")))
            rmse_era5.append(float(m.group("rmse_era5")))
            rmse_naive.append(float(m.group("rmse_naive")))
            rmse_trans.append(float(m.group("rmse_trans")))
            acc_era5.append(float(m.group("acc_era5")))
            acc_naive.append(float(m.group("acc_naive")))
            acc_trans.append(float(m.group("acc_trans")))
    return {
        "steps": steps,
        "rmse_era5": rmse_era5,
        "rmse_naive": rmse_naive,
        "rmse_trans": rmse_trans,
        "acc_era5": acc_era5,
        "acc_naive": acc_naive,
        "acc_trans": acc_trans,
    }

def load_series(metrics_root, tag, date):
    txt_path = Path(metrics_root) / tag / date / f"metrics_{tag}_{date}.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"缺少文件: {txt_path}")
    return parse_metrics_txt(txt_path)

def average_metrics(metrics_root, tag, dates):
    all_metrics = []
    for date in dates:
        try:
            data = load_series(metrics_root, tag, date)
            all_metrics.append(data)
        except Exception as e:
            print(f"跳过 {tag} {date}: {e}")
    if not all_metrics:
        raise RuntimeError(f"没有可用的 {tag} 数据")
    steps = all_metrics[0]["steps"]
    n = len(steps)
    def stack_and_mean(key):
        arr = np.stack([m[key] for m in all_metrics if len(m[key])==n])
        return arr.mean(axis=0)
    return {
        "steps": steps,
        "rmse_era5": stack_and_mean("rmse_era5"),
        "rmse_naive": stack_and_mean("rmse_naive"),
        "rmse_trans": stack_and_mean("rmse_trans"),
        "acc_era5": stack_and_mean("acc_era5"),
        "acc_naive": stack_and_mean("acc_naive"),
        "acc_trans": stack_and_mean("acc_trans"),
    }

def _truncate_series(series, num_steps=None):
    if num_steps is None:
        return series
    n = min(int(num_steps), len(series["steps"]))
    out = {}
    for k, v in series.items():
        if isinstance(v, (list, np.ndarray)):
            out[k] = np.array(v)[:n]
        else:
            out[k] = v
    return out


def plot_curve(avg_by_tag, output_dir, metric="acc", num_steps=None, title_suffix="z500"):
    if len(avg_by_tag) == 0:
        raise RuntimeError("没有可绘制的 tag 数据")

    # 以第一个 tag 作为步长参考
    first_tag = next(iter(avg_by_tag))
    ref = _truncate_series(avg_by_tag[first_tag], num_steps=num_steps)
    steps = ref["steps"]

    plt.figure(figsize=(10, 5))
    # 预留给模型曲线的颜色（不使用 gray，避免和 Naive ERA5 冲突）
    model_colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:olive",
        "tab:cyan",
    ]

    # 仅绘制 --tags 中的模型曲线
    if metric == "acc":
        for idx, (tag, data) in enumerate(avg_by_tag.items()):
            d = _truncate_series(data, num_steps=num_steps)
            if len(d["steps"]) != len(steps):
                print(f"跳过 {tag}: step 数量与参考不一致")
                continue
            c = model_colors[idx % len(model_colors)]
            plt.plot(steps, d["acc_trans"], label=tag, marker="o", color=c)
        plt.xlabel("Forecast Step")
        plt.ylabel("ACC")
        plt.title(f"平均ACC ({title_suffix})")
        plt.ylim(0.98, 1.00)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "acc_curve_avg.png"))
        plt.close()

    elif metric == "rmse":
        for idx, (tag, data) in enumerate(avg_by_tag.items()):
            d = _truncate_series(data, num_steps=num_steps)
            if len(d["steps"]) != len(steps):
                print(f"跳过 {tag}: step 数量与参考不一致")
                continue
            c = model_colors[idx % len(model_colors)]
            plt.plot(steps, d["rmse_trans"], label=tag, marker="o", color=c)
        plt.xlabel("Forecast Step")
        plt.ylabel("RMSE")
        plt.title(f"平均RMSE ({title_suffix})")
        plt.ylim(50, 160)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rmse_curve_avg.png"))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot averaged ACC/RMSE curves from metrics txt")
    parser.add_argument("--metrics_root", type=str, default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/infertest/metrics")
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=["3yr", "3yr_L1+Gradloss", "3yr_L2_NS", "3yr_Charbonnier"],
        help="要对比的模型 tag 列表，例如: 3yr_L2_NS 3yr_Charbonnier",
    )
    parser.add_argument(
        "--dates",
        type=str,
        nargs="+",
        default=["20250101", "20250115", "20250131", "20250214", "20250301", "20250315", "20250331", "20250501", "20250515", "20250601"],
    )
    parser.add_argument("--num_steps", type=int, default=8, help="只绘制前 N 个 forecast step")
    parser.add_argument("--title_suffix", type=str, default="z500", help="图标题后缀")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.metrics_root
    os.makedirs(output_dir, exist_ok=True)

    avg_by_tag = {}
    for tag in args.tags:
        avg_by_tag[tag] = average_metrics(args.metrics_root, tag, args.dates)

    plot_curve(avg_by_tag, output_dir, metric="acc", num_steps=args.num_steps, title_suffix=args.title_suffix)
    plot_curve(avg_by_tag, output_dir, metric="rmse", num_steps=args.num_steps, title_suffix=args.title_suffix)

if __name__ == "__main__":
    main()
