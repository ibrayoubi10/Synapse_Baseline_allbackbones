# author: Ibrahim (refactor)
import numpy as np
from medpy import metric as medpy_metric
import math

def calculate_metric_percase(pred, gt):
    """
    pred, gt: binary masks (H,W) or booleans
    returns: dice, jaccard, hd95 (floats or np.nan)
    """
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    if gt.sum() == 0:
        return np.nan, np.nan, np.nan
    if pred.sum() == 0:
        return 0.0, 0.0, np.nan
    dice = medpy_metric.binary.dc(pred, gt)
    jac = medpy_metric.binary.jc(pred, gt)
    try:
        hd95 = medpy_metric.binary.hd95(pred, gt)
    except Exception:
        hd95 = np.nan
    return float(dice), float(jac), float(hd95)

def multilabel_metric(pred, gt, num_classes):
    """
    pred: 2D int map
    gt: 2D int map
    returns list of tuples for classes 1..num_classes-1
    """
    out = []
    for c in range(1, num_classes):
        out.append(calculate_metric_percase(pred == c, gt == c))
    return out

# Running accumulators helpers
def init_running_metrics(num_classes):
    running = {
        "per_class": [ [] for _ in range(num_classes) ],  # index c -> list of (dice,jac,hd95) for that class
        "n_samples": 0
    }
    return running

def update_running_metrics(running, pred, gt, num_classes):
    """
    pred, gt: 2D arrays (H,W) of ints
    """
    running["n_samples"] += 1
    metrics = multilabel_metric(pred, gt, num_classes)
    for idx, m in enumerate(metrics, start=1):
        running["per_class"][idx].append(m)  # append tuple (dice,jac,hd95)
    return running

def finalize_metrics(running, num_classes):
    """
    returns:
      per_class: dict index->(mean_dice, mean_jac, mean_hd95) (NaN safe)
      macro: (mean_dice, mean_jac, mean_hd95) computed across classes ignoring NaNs
    """
    per_class = {}
    dices = []
    jacs = []
    hd95s = []
    for c in range(num_classes):
        arr = running["per_class"][c]
        if len(arr) == 0:
            per_class[c] = (np.nan, np.nan, np.nan)
            continue
        arr_np = np.array(arr, dtype=float)  # shape (N,3)
        d = np.nanmean(arr_np[:,0])
        j = np.nanmean(arr_np[:,1])
        h = np.nanmean(arr_np[:,2])
        per_class[c] = (float(d) if not math.isnan(d) else np.nan,
                        float(j) if not math.isnan(j) else np.nan,
                        float(h) if not math.isnan(h) else np.nan)
        if c != 0:  # exclude background for macro
            if not math.isnan(d): dices.append(d)
            if not math.isnan(j):  jacs.append(j)
            if not math.isnan(h):  hd95s.append(h)
    macro = (
        float(np.nanmean(dices)) if len(dices)>0 else np.nan,
        float(np.nanmean(jacs)) if len(jacs)>0 else np.nan,
        float(np.nanmean(hd95s)) if len(hd95s)>0 else np.nan
    )
    return per_class, macro