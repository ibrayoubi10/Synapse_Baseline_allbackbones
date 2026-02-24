import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
CSV_PATH = "CSVs/history_ResNet.csv"  
OUT_DIR = "plots_ResNet"
DPI = 170
ROLL = 1  # mets 5 ou 7 si tu as beaucoup d'epochs pour lisser

os.makedirs(OUT_DIR, exist_ok=True)

# ======================
# LOAD
# ======================
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

# numeric conversion (safe)
for c in df.columns:
    if df[c].dtype == object:
        df[c] = (df[c].astype(str)
                      .str.replace(",", ".", regex=False)
                      .str.strip())
    df[c] = pd.to_numeric(df[c], errors="coerce")

x_col = "epoch" if "epoch" in df.columns else df.columns[0]
x = df[x_col].to_numpy()

def rmean(y, w=ROLL):
    if w <= 1:
        return y
    return pd.Series(y).rolling(window=w, min_periods=max(1, w//2)).mean().to_numpy()

def save_plot(title, xlabel, ylabel, path):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI)
    plt.close()

def plot_train_test(col_train, col_test, title, ylabel, fname):
    if col_train not in df.columns or col_test not in df.columns:
        return
    plt.figure()
    plt.plot(x, rmean(df[col_train].to_numpy()), label=col_train)
    plt.plot(x, rmean(df[col_test].to_numpy()), label=col_test)
    plt.legend(loc="best", fontsize=8)
    save_plot(title, x_col, ylabel, os.path.join(OUT_DIR, fname))

def plot_single(col, title, ylabel, fname):
    if col not in df.columns:
        return
    plt.figure()
    plt.plot(x, rmean(df[col].to_numpy()), label=col)
    plt.legend(loc="best", fontsize=8)
    save_plot(title, x_col, ylabel, os.path.join(OUT_DIR, fname))

def find_class_cols(prefix, metric):
    # example: train_dice_c01, test_hd95_c08, etc.
    pat = re.compile(rf"^{prefix}_{metric}_c(\d+)$")
    cols = []
    for c in df.columns:
        m = pat.match(c)
        if m:
            cols.append((int(m.group(1)), c))
    cols.sort(key=lambda t: t[0])
    return cols

def plot_per_class(prefix, metric, title, ylabel, fname):
    cols = find_class_cols(prefix, metric)
    if not cols:
        return
    plt.figure()
    for k, c in cols:
        plt.plot(x, rmean(df[c].to_numpy()), label=f"c{k:02d}")
    plt.legend(loc="best", fontsize=7, ncol=2)
    save_plot(title, x_col, ylabel, os.path.join(OUT_DIR, fname))

def plot_gap_per_class(metric, title, ylabel, fname):
    # gap = train - test for each class
    train_cols = dict(find_class_cols("train", metric))
    test_cols  = dict(find_class_cols("test", metric))
    common = sorted(set(train_cols.keys()) & set(test_cols.keys()))
    if not common:
        return
    plt.figure()
    for k in common:
        gap = df[train_cols[k]].to_numpy() - df[test_cols[k]].to_numpy()
        plt.plot(x, rmean(gap), label=f"c{k:02d}")
    plt.legend(loc="best", fontsize=7, ncol=2)
    save_plot(title, x_col, ylabel, os.path.join(OUT_DIR, fname))

# ======================
# SUMMARY (best / last)
# ======================
def safe_argmax(col):
    if col not in df.columns:
        return None
    s = df[col].astype(float)
    if s.notna().sum() == 0:
        return None
    return int(s.idxmax())

best_idx = safe_argmax("test_macro_dice")
last_idx = len(df) - 1

def row_report(i, name):
    cols = [
        "epoch", "lr",
        "train_loss", "test_loss",
        "train_macro_dice", "test_macro_dice",
        "train_macro_jac", "test_macro_jac",
        "train_macro_hd95", "test_macro_hd95",
    ]
    cols = [c for c in cols if c in df.columns]
    r = df.loc[i, cols]
    print(f"\n=== {name} (row={i}) ===")
    print(r.to_string())

row_report(last_idx, "LAST")
if best_idx is not None:
    row_report(best_idx, "BEST by test_macro_dice")

# ======================
# 1) LR
# ======================
plot_single("lr", "Learning Rate", "lr", "01_lr.png")

# ======================
# 2) LOSSES (train vs test)
# ======================
plot_train_test("train_loss", "test_loss", "Total Loss (train vs test)", "loss", "02_loss_total.png")
plot_train_test("train_ce", "test_ce", "Cross-Entropy (train vs test)", "ce", "03_loss_ce.png")
plot_train_test("train_dice_loss", "test_dice_loss", "Dice Loss (train vs test)", "dice_loss", "04_loss_dice.png")

# ======================
# 3) MACRO METRICS (train vs test)
# ======================
plot_train_test("train_macro_dice", "test_macro_dice", "Macro Dice (train vs test)", "dice", "05_macro_dice.png")
plot_train_test("train_macro_jac", "test_macro_jac", "Macro Jaccard/IoU (train vs test)", "jaccard", "06_macro_jaccard.png")
plot_train_test("train_macro_hd95", "test_macro_hd95", "Macro HD95 (train vs test)", "hd95", "07_macro_hd95.png")

# ======================
# 4) PER-CLASS CURVES
# ======================
plot_per_class("train", "dice", "Train Dice per class", "dice", "08_train_dice_per_class.png")
plot_per_class("test",  "dice", "Test Dice per class",  "dice", "09_test_dice_per_class.png")

plot_per_class("train", "jac",  "Train Jaccard per class", "jaccard", "10_train_jaccard_per_class.png")
plot_per_class("test",  "jac",  "Test Jaccard per class",  "jaccard", "11_test_jaccard_per_class.png")

plot_per_class("train", "hd95", "Train HD95 per class", "hd95", "12_train_hd95_per_class.png")
plot_per_class("test",  "hd95", "Test HD95 per class",  "hd95", "13_test_hd95_per_class.png")

# ======================
# 5) TRAIN–TEST GAP (overfitting signal)
# ======================
plot_gap_per_class("dice", "Gap (train - test) Dice per class", "dice gap", "14_gap_dice_per_class.png")
plot_gap_per_class("jac",  "Gap (train - test) Jaccard per class", "jaccard gap", "15_gap_jaccard_per_class.png")
plot_gap_per_class("hd95", "Gap (train - test) HD95 per class", "hd95 gap", "16_gap_hd95_per_class.png")

print(f"\nSaved plots -> {OUT_DIR}/ (files: {len(os.listdir(OUT_DIR))})")