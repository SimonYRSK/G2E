import argparse
import os
import re
from pathlib import Path
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


def main():
    parser = argparse.ArgumentParser(description="Plot ACC/RMSE curves from metrics txt")
    parser.add_argument("--metrics_root", type=str, default="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/infertest/metrics")
    parser.add_argument("--dates", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.metrics_root
    os.makedirs(output_dir, exist_ok=True)

    for date in args.dates:
        data_1yr = load_series(args.metrics_root, "1yr", date)
        data_3yr = load_series(args.metrics_root, "3yr", date)

        steps = data_1yr["steps"]

        plt.figure(figsize=(10, 5))
        plt.plot(steps, data_1yr["acc_era5"], label="Naive ERA5", marker="o")
        plt.plot(steps, data_1yr["acc_naive"], label="Naive GFS", marker="o")
        plt.plot(steps, data_1yr["acc_trans"], label="GFS2ERA5 1yr", marker="o")
        plt.plot(steps, data_3yr["acc_trans"], label="GFS2ERA5 3yr", marker="o")
        plt.xlabel("Forecast Step")
        plt.ylabel("ACC")
        plt.title(f"ACC (z500) {date}")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"acc_curve_{date}.png"))
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(steps, data_1yr["rmse_era5"], label="Naive ERA5", marker="o")
        plt.plot(steps, data_1yr["rmse_naive"], label="Naive GFS", marker="o")
        plt.plot(steps, data_1yr["rmse_trans"], label="GFS2ERA5 1yr", marker="o")
        plt.plot(steps, data_3yr["rmse_trans"], label="GFS2ERA5 3yr", marker="o")
        plt.xlabel("Forecast Step")
        plt.ylabel("RMSE")
        plt.title(f"RMSE (z500) {date}")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"rmse_curve_{date}.png"))
        plt.close()


if __name__ == "__main__":
    main()
