# plot_history.py
# Creates:
#   Image 1: loss curves (train vs test)
#   Image 2: macro metrics curves (Dice, JAC, HD95) for train vs test
#   Image 3: per-class Dice curves (Train-only + Test-only) in one figure
#
# Usage:
#   python plot_history.py
# or:
#   python plot_history.py --csv /path/to/history.csv --out_dir /path/to/out

"""
terminal command to run Unet model: 

python3 /data3/nkozah/my_project/Ibrahim/plot/plots.py --csv /data3/nkozah/my_project/Ibrahim/experiments/Synapse_UNet_2D/history.csv --out_dir /data3/nkozah/my_project/Ibrahim/plot
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = (
                out[c].astype(str)
                .str.replace(",", ".", regex=False)
                .str.strip()
            )
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _roll(y: np.ndarray, w: int) -> np.ndarray:
    if w is None or w <= 1:
        return y
    s = pd.Series(y)
    return s.rolling(window=w, min_periods=1, center=False).mean().to_numpy()


def _savefig(path: str, dpi: int = 180):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_loss(df: pd.DataFrame, x: np.ndarray, out_path: str, roll: int):
    required = ["train_loss", "test_loss"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' in CSV")

    plt.figure(figsize=(10, 5))
    plt.plot(x, _roll(df["train_loss"].to_numpy(), roll), label="train_loss")
    plt.plot(x, _roll(df["test_loss"].to_numpy(), roll), label="test_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss curves (Train vs Test)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _savefig(out_path)


def plot_macro_metrics(df: pd.DataFrame, x: np.ndarray, out_path: str, roll: int):
    needed = [
        ("train_macro_dice", "test_macro_dice", "Macro Dice"),
        ("train_macro_jac",  "test_macro_jac",  "Macro Jaccard"),
        ("train_macro_hd95", "test_macro_hd95", "Macro HD95"),
    ]
    for a, b, _ in needed:
        if a not in df.columns or b not in df.columns:
            raise KeyError(f"Missing columns '{a}' or '{b}' in CSV")

    # One "image" containing 3 plots (Dice/JAC/HD95)
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12), sharex=True)

    for ax, (tr, te, title) in zip(axes, needed):
        ax.plot(x, _roll(df[tr].to_numpy(), roll), label=tr)
        ax.plot(x, _roll(df[te].to_numpy(), roll), label=te)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Epoch")
    fig.suptitle("Macro metrics (Train vs Test)", y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def _find_perclass_dice_cols(df: pd.DataFrame, prefix: str):
    # prefix: "train_dice_c" or "test_dice_c"
    cols = []
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    for c in df.columns:
        m = pat.match(c)
        if m:
            cols.append((int(m.group(1)), c))
    cols.sort(key=lambda t: t[0])
    return [c for _, c in cols]


def plot_perclass_dice(df: pd.DataFrame, x: np.ndarray, out_path: str, roll: int):
    tr_cols = _find_perclass_dice_cols(df, "train_dice_c")
    te_cols = _find_perclass_dice_cols(df, "test_dice_c")

    if len(tr_cols) == 0:
        raise KeyError("No per-class train dice columns found (expected like train_dice_c01, ...).")
    if len(te_cols) == 0:
        raise KeyError("No per-class test dice columns found (expected like test_dice_c01, ...).")

    # One "image" containing 2 plots: train per-class, test per-class
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)

    ax = axes[0]
    for c in tr_cols:
        ax.plot(x, _roll(df[c].to_numpy(), roll), label=c)
    ax.set_title("Per-class Dice (Train)")
    ax.set_ylabel("Dice")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)

    ax = axes[1]
    for c in te_cols:
        ax.plot(x, _roll(df[c].to_numpy(), roll), label=c)
    ax.set_title("Per-class Dice (Test)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)

    fig.suptitle("Per-class Dice curves", y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="history.csv")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--roll", type=int, default=1, help="rolling mean window (1 disables)")
    args = ap.parse_args()

    csv_path = os.path.abspath(args.csv)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(csv_path), "plots")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = _to_numeric(df)

    x_col = "epoch" if "epoch" in df.columns else df.columns[0]
    x = df[x_col].to_numpy()

    plot_loss(df, x, os.path.join(out_dir, "Image1_loss_train_vs_test.png"), args.roll)
    plot_macro_metrics(df, x, os.path.join(out_dir, "Image2_macro_metrics_train_vs_test.png"), args.roll)
    plot_perclass_dice(df, x, os.path.join(out_dir, "Image3_perclass_dice_train_and_test.png"), args.roll)

    print("Saved plots to:", out_dir)
    print(" - Image1_loss_train_vs_test.png")
    print(" - Image2_macro_metrics_train_vs_test.png")
    print(" - Image3_perclass_dice_train_and_test.png")


if __name__ == "__main__":
    main()