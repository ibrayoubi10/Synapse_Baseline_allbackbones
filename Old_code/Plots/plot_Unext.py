import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
CSV_PATH = "CSVs/history_unext.csv"  # <- change if needed (absolute path recommended on Windows)
OUT_DIR = "plots_unext"
DPI = 180
ROLL = 1  # 5/7 si beaucoup d'epochs pour lisser

os.makedirs(OUT_DIR, exist_ok=True)

# ======================
# LOAD + CLEAN
# ======================
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

for c in df.columns:
    if df[c].dtype == object:
        df[c] = (df[c].astype(str)
                      .str.replace(",", ".", regex=False)
                      .str.strip())
    df[c] = pd.to_numeric(df[c], errors="coerce")

# choose x axis
if "epoch" in df.columns and df["epoch"].notna().sum() > 0:
    x_col = "epoch"
else:
    df["_index"] = np.arange(len(df))
    x_col = "_index"

x = df[x_col].to_numpy()

def rmean(y, w=ROLL):
    if w <= 1:
        return y
    return pd.Series(y).rolling(window=w, min_periods=max(1, w//2)).mean().to_numpy()

def save_and_show(title, xlabel, ylabel, filename):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, filename)
    plt.savefig(out, dpi=DPI)
    plt.show()
    plt.close()

def plot_train_test(col_train, col_test, title, ylabel, filename):
    if col_train not in df.columns or col_test not in df.columns:
        return False
    plt.figure()
    plt.plot(x, rmean(df[col_train].to_numpy()), label=col_train)
    plt.plot(x, rmean(df[col_test].to_numpy()), label=col_test)
    plt.legend(loc="best", fontsize=8)
    save_and_show(title, x_col, ylabel, filename)
    return True

def plot_single(col, title, ylabel, filename):
    if col not in df.columns:
        return False
    plt.figure()
    plt.plot(x, rmean(df[col].to_numpy()), label=col)
    plt.legend(loc="best", fontsize=8)
    save_and_show(title, x_col, ylabel, filename)
    return True

def find_class_cols(prefix, metric):
    # expects: train_dice_c01, test_jac_c08, train_hd95_c03, ...
    pat = re.compile(rf"^{prefix}_{metric}_c(\d+)$")
    cols = []
    for c in df.columns:
        m = pat.match(c)
        if m:
            cols.append((int(m.group(1)), c))
    cols.sort(key=lambda t: t[0])
    return cols

def plot_per_class(prefix, metric, title, ylabel, filename):
    cols = find_class_cols(prefix, metric)
    if not cols:
        return False
    plt.figure()
    for k, c in cols:
        plt.plot(x, rmean(df[c].to_numpy()), label=f"c{k:02d}")
    plt.legend(loc="best", fontsize=7, ncol=2)
    save_and_show(title, x_col, ylabel, filename)
    return True

def plot_gap_per_class(metric, title, ylabel, filename):
    train_cols = dict(find_class_cols("train", metric))
    test_cols  = dict(find_class_cols("test", metric))
    common = sorted(set(train_cols.keys()) & set(test_cols.keys()))
    if not common:
        return False
    plt.figure()
    for k in common:
        gap = df[train_cols[k]].to_numpy() - df[test_cols[k]].to_numpy()
        plt.plot(x, rmean(gap), label=f"c{k:02d}")
    plt.legend(loc="best", fontsize=7, ncol=2)
    save_and_show(title, x_col, ylabel, filename)
    return True

# ======================
# QUICK SUMMARY
# ======================
def safe_best(col):
    if col not in df.columns or df[col].notna().sum() == 0:
        return None
    i = int(df[col].astype(float).idxmax())
    return i, df.loc[i, "epoch"] if "epoch" in df.columns else i, float(df.loc[i, col])

best = safe_best("test_macro_dice")
if best:
    i, ep, v = best
    print(f"[BEST] test_macro_dice = {v:.4f} at row={i} (epoch={ep})")
print(f"[LAST] row={len(df)-1}")

# ======================
# PLOTS (core)
# ======================
plot_single("lr", "Learning Rate", "lr", "01_lr.png")

# plot_train_test("train_loss", "test_loss", "Total Loss (train vs test)", "loss", "02_loss_total.png")
#plot_train_test("train_ce", "test_ce", "Cross-Entropy (train vs test)", "ce", "03_loss_ce.png")
# plot_train_test("train_dice_loss", "test_dice_loss", "Dice Loss (train vs test)", "dice_loss", "04_loss_dice.png")

plot_train_test("macro_dice", "test_macro_dice", "Macro Dice (train vs test)", "dice", "05_macro_dice.png")
plot_train_test("macro_jac", "test_macro_jac", "Macro Jaccard/IoU (train vs test)", "jaccard", "06_macro_jaccard.png")
plot_train_test("macro_hd95", "test_macro_hd95", "Macro HD95 (train vs test)", "hd95", "07_macro_hd95.png")

# ======================
# PLOTS (per-class)
# ======================
plot_per_class("train", "dice", "Train Dice per class", "dice", "08_train_dice_per_class.png")
plot_per_class("test",  "dice", "Test Dice per class",  "dice", "09_test_dice_per_class.png")

plot_per_class("train", "jac",  "Train Jaccard per class", "jaccard", "10_train_jaccard_per_class.png")
plot_per_class("test",  "jac",  "Test Jaccard per class",  "jaccard", "11_test_jaccard_per_class.png")

plot_per_class("train", "hd95", "Train HD95 per class", "hd95", "12_train_hd95_per_class.png")
plot_per_class("test",  "hd95", "Test HD95 per class",  "hd95", "13_test_hd95_per_class.png")

#plot_gap_per_class("dice", "Gap (train - test) Dice per class", "dice gap", "14_gap_dice_per_class.png")
# plot_gap_per_class("jac",  "Gap (train - test) Jaccard per class", "jaccard gap", "15_gap_jaccard_per_class.png")
# plot_gap_per_class("hd95", "Gap (train - test) HD95 per class", "hd95 gap", "16_gap_hd95_per_class.png")

print(f"\nSaved PNGs to: {os.path.abspath(OUT_DIR)}")
print(f"Files: {len(os.listdir(OUT_DIR))}")