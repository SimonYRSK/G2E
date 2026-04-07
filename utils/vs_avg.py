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

def plot_curve(avg_1yr, avg_3yr, output_dir, metric="acc"):
    steps = avg_1yr["steps"]
    plt.figure(figsize=(10, 5))
    if metric == "acc":
        plt.plot(steps, avg_1yr["acc_era5"], label="Naive ERA5", marker="o", color="tab:blue")
        plt.plot(steps, avg_1yr["acc_naive"], label="Naive GFS", marker="o", color="gray")
        plt.plot(steps, avg_1yr["acc_trans"], label="L2", marker="o", color="tab:orange")
        plt.plot(steps, avg_3yr["acc_trans"], label="L1+GRAD", marker="o", color="tab:green")
        acc_1yr = np.array(avg_1yr["acc_trans"])
        acc_3yr = np.array(avg_3yr["acc_trans"])
        steps_arr = np.array(steps)
        mask_blue = acc_3yr > acc_1yr 
        if np.any(mask_blue):
            plt.fill_between(steps_arr, acc_1yr, acc_3yr, where=mask_blue, color="blue", alpha=0.2, label="L1+GRAD > L2")
        mask_red = acc_1yr > acc_3yr
        if np.any(mask_red):
            plt.fill_between(steps_arr, acc_3yr, acc_1yr, where=mask_red, color="red", alpha=0.2, label="L2 > L1+GRAD")
        plt.xlabel("Forecast Step")
        plt.ylabel("ACC")
        plt.title("平均ACC (z500)")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "acc_curve_avg.png"))
        plt.close()
    elif metric == "rmse":
        plt.plot(steps, avg_1yr["rmse_era5"], label="Naive ERA5", marker="o", color="tab:blue")
        plt.plot(steps, avg_1yr["rmse_naive"], label="Naive GFS", marker="o", color="gray")
        plt.plot(steps, avg_1yr["rmse_trans"], label="L2", marker="o", color="tab:orange")
        plt.plot(steps, avg_3yr["rmse_trans"], label="L1+GRAD", marker="o", color="tab:green")
        rmse_1yr = np.array(avg_1yr["rmse_trans"])
        rmse_3yr = np.array(avg_3yr["rmse_trans"])
        steps_arr = np.array(steps)
        mask_blue = rmse_3yr < rmse_1yr
        if np.any(mask_blue):
            plt.fill_between(steps_arr, rmse_3yr, rmse_1yr, where=mask_blue, color="blue", alpha=0.2, label="L1+GRAD < L2")
        mask_red = rmse_1yr < rmse_3yr
        if np.any(mask_red):
            plt.fill_between(steps_arr, rmse_1yr, rmse_3yr, where=mask_red, color="red", alpha=0.2, label="L2 < L1+GRAD")
        plt.xlabel("Forecast Step")
        plt.ylabel("RMSE")
        plt.title("平均RMSE (z500)")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rmse_curve_avg.png"))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot averaged ACC/RMSE curves from metrics txt")
    parser.add_argument("--metrics_root", type=str, default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/infertest/metrics")
    parser.add_argument(
        "--dates",
        type=str,
        nargs="+",
        default=["20250101", "20250115", "20250131", "20250214", "20250301", "20250315", "20250331", "20250501", "20250515", "20250601"],
    )
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.metrics_root
    os.makedirs(output_dir, exist_ok=True)

    avg_1yr = average_metrics(args.metrics_root, "3yr", args.dates)
    avg_3yr = average_metrics(args.metrics_root, "3yr_L1+Gradloss", args.dates)

    plot_curve(avg_1yr, avg_3yr, output_dir, metric="acc")
    plot_curve(avg_1yr, avg_3yr, output_dir, metric="rmse")

if __name__ == "__main__":
    main()
