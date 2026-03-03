# main_efficientunet_2d_testvol.py
#
# SAME structure as your main_deeplabv3plus_2d_testvol.py, but:
#   - Model = EfficientUNet_EfficientNet (your multilabel/multiclass logits model)
#   - Uses EfficientNet.encoder() under the hood
#
# Assumes you have:
#   Ibrahim/networks/EfficientUNet_EfficientNet_MultiLabel.py  (your code)
#   Ibrahim/networks/EfficientNet.py                          (your EfficientNet implementation)
#
# Run example:
#   python main_efficientunet_2d_testvol.py --effnet efficientnet-b2 --encoder_pretrained 1 --concat_input 1

import os
import sys
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from scipy.ndimage import zoom

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Ibrahim.Datasets.synapse_dataset_config import SynapseDataset, RandomGenerator
from Ibrahim.losses.DiceLoss import MultiClassDiceLoss
from Ibrahim.metrics.Multilabel_metrics import init_running_metrics, update_running_metrics, finalize_metrics

from Ibrahim.networks.EfficientUnet_MultiLabel import (
    get_efficientunet_multilabel_b0,
    get_efficientunet_multilabel_b1,
    get_efficientunet_multilabel_b2,
    get_efficientunet_multilabel_b3,
    get_efficientunet_multilabel_b4,
    get_efficientunet_multilabel_b5,
    get_efficientunet_multilabel_b6,
    get_efficientunet_multilabel_b7,
)


def estimate_ce_weights_from_loader(train_loader, num_classes, device, max_batches=200,
                                    clamp_min=0.1, clamp_max=10.0):
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for k, s in enumerate(train_loader):
        if k >= max_batches:
            break
        y = s["label"].view(-1)
        counts += torch.bincount(y, minlength=num_classes).double()
    freq = counts / counts.sum().clamp_min(1.0)
    w = 1.0 / (freq + 1e-12)
    w = (w / w.mean().clamp_min(1e-12)).float()
    w = torch.clamp(w, clamp_min, clamp_max).to(device)
    return w, counts


def append_csv(csv_path, header, row):
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", encoding="utf-8") as f:
        if not exists:
            f.write(header)
        f.write(row)


def _prep_2d_batch(images, labels, device):
    if images.dim() == 3:
        images = images.unsqueeze(1)
    imgs = images.to(device, dtype=torch.float32, non_blocking=True)
    if imgs.shape[1] == 1:
        imgs = imgs.repeat(1, 3, 1, 1)
    gts = labels.to(device, dtype=torch.long, non_blocking=True)
    return imgs, gts


def _ensure_depth_first(vol_np):
    if vol_np.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {vol_np.shape}")
    depth_axis = int(np.argmin(list(vol_np.shape)))
    if depth_axis != 0:
        vol_np = np.moveaxis(vol_np, depth_axis, 0)
    return vol_np


def _nan_to_empty(x):
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


def _per_class_to_arrays(per_class, C):
    dice = [np.nan] * C
    jac = [np.nan] * C
    hd95 = [np.nan] * C

    if isinstance(per_class, dict):
        for k, v in per_class.items():
            try:
                cid = int(k)
            except Exception:
                continue
            if cid < 0 or cid >= C:
                continue
            if isinstance(v, (list, tuple)) and len(v) >= 3:
                dice[cid], jac[cid], hd95[cid] = v[0], v[1], v[2]
            elif isinstance(v, dict):
                dice[cid] = v.get("dice", np.nan)
                jac[cid] = v.get("jac", np.nan)
                hd95[cid] = v.get("hd95", np.nan)

    elif isinstance(per_class, (list, tuple)):
        for cid in range(min(C, len(per_class))):
            v = per_class[cid]
            if isinstance(v, (list, tuple)) and len(v) >= 3:
                dice[cid], jac[cid], hd95[cid] = v[0], v[1], v[2]
            elif isinstance(v, dict):
                dice[cid] = v.get("dice", np.nan)
                jac[cid] = v.get("jac", np.nan)
                hd95[cid] = v.get("hd95", np.nan)

    return dice, jac, hd95


class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4, mode="max"):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        assert mode in ("max", "min")
        self.mode = mode
        self.best = None
        self.bad_epochs = 0

    def _is_improvement(self, current):
        if self.best is None:
            return True
        if self.mode == "max":
            return current > (self.best + self.min_delta)
        return current < (self.best - self.min_delta)

    def step(self, current):
        if current is None or (isinstance(current, float) and np.isnan(current)):
            self.bad_epochs += 1
            return False, (self.bad_epochs >= self.patience)
        current = float(current)
        if self._is_improvement(current):
            self.best = current
            self.bad_epochs = 0
            return True, False
        self.bad_epochs += 1
        return False, (self.bad_epochs >= self.patience)


class TestSliceTransform:
    def __init__(self, output_size):
        self.output_size = tuple(output_size)

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        x, y = image.shape
        if (x, y) != self.output_size:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.int64))
        return {"image": image, "label": label}


class TestVol2DSliceDataset(Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base = base_dataset
        self.transform = transform
        self.index = []
        for case_idx in range(len(self.base)):
            s = self.base[case_idx]
            img = _ensure_depth_first(np.asarray(s["image"]))
            lab = _ensure_depth_first(np.asarray(s["label"]))
            if img.shape[0] != lab.shape[0]:
                raise ValueError(f"Depth mismatch: {img.shape} vs {lab.shape}")
            for z in range(img.shape[0]):
                self.index.append((case_idx, z))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        case_idx, z = self.index[i]
        s = self.base[case_idx]
        img = _ensure_depth_first(np.asarray(s["image"]))
        lab = _ensure_depth_first(np.asarray(s["label"]))
        sample = {"image": img[z], "label": lab[z]}
        if self.transform is not None:
            t = self.transform(sample)
            sample["image"], sample["label"] = t["image"], t["label"]
        return sample


def evaluate_on_test_2d(model, test_loader, num_classes, device, ce_loss, dice_loss):
    model.eval()
    running = init_running_metrics(num_classes)
    loss_sum = ce_sum = dice_sum = 0.0
    n_batches = 0

    with torch.no_grad():
        for samples in tqdm(test_loader, total=len(test_loader), desc="TEST"):
            imgs, gts = _prep_2d_batch(samples["image"], samples["label"], device)
            logits = model(imgs)

            loss_ce = ce_loss(logits, gts)
            loss_d = dice_loss(logits, gts)
            loss = loss_ce + loss_d

            loss_sum += float(loss.detach().cpu())
            ce_sum += float(loss_ce.detach().cpu())
            dice_sum += float(loss_d.detach().cpu())
            n_batches += 1

            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy()
            gt = gts.cpu().numpy()
            for b in range(pred.shape[0]):
                running = update_running_metrics(running, pred[b], gt[b], num_classes)

    per_class, macro = finalize_metrics(running, num_classes)
    return per_class, macro, (
        loss_sum / max(1, n_batches),
        ce_sum / max(1, n_batches),
        dice_sum / max(1, n_batches),
    )


def get_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", type=str, default="/data3/nkozah/my_project/Data/synapse")
    p.add_argument("--num_classes", type=int, default=9)
    p.add_argument("--NB_EPOCH", type=int, default=200)
    p.add_argument("--LR", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--test_batch_size", type=int, default=8)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--exp_name", type=str, default="Synapse_EfficientUNet_2D_testvol")
    p.add_argument("--metrics_every", type=int, default=20)
    p.add_argument("--save_best_on", type=str, default="macro_dice", choices=["macro_dice", "loss"])
    p.add_argument("--random_seed", type=int, default=1234)
    p.add_argument("--ce_weight_batches", type=int, default=200)
    p.add_argument("--ce_w_min", type=float, default=0.1)
    p.add_argument("--ce_w_max", type=float, default=10.0)

    # EfficientNet variant + pretrained + concat_input
    p.add_argument("--effnet", type=str, default="efficientnet-b2",
                   choices=["efficientnet-b0","efficientnet-b1","efficientnet-b2","efficientnet-b3",
                            "efficientnet-b4","efficientnet-b5","efficientnet-b6","efficientnet-b7"])
    p.add_argument("--encoder_pretrained", type=int, default=1, choices=[0, 1])
    p.add_argument("--concat_input", type=int, default=1, choices=[0, 1])

    p.add_argument("--early_stop", type=int, default=1, choices=[0, 1])
    p.add_argument("--es_patience", type=int, default=25)
    p.add_argument("--es_min_delta", type=float, default=1e-4)
    p.add_argument("--out_root", type=str, default="/data3/nkozah/my_project/Ibrahim/experiments")
    p.add_argument("--num_workers", type=int, default=0)  # IMPORTANT
    p.add_argument("--pin_memory", type=int, default=1, choices=[0, 1])
    return p


def _build_model(effnet_name: str, num_classes: int, pretrained: bool, concat_input: bool) -> nn.Module:
    effnet_name = effnet_name.lower().strip()
    builders = {
        "efficientnet-b0": get_efficientunet_multilabel_b0,
        "efficientnet-b1": get_efficientunet_multilabel_b1,
        "efficientnet-b2": get_efficientunet_multilabel_b2,
        "efficientnet-b3": get_efficientunet_multilabel_b3,
        "efficientnet-b4": get_efficientunet_multilabel_b4,
        "efficientnet-b5": get_efficientunet_multilabel_b5,
        "efficientnet-b6": get_efficientunet_multilabel_b6,
        "efficientnet-b7": get_efficientunet_multilabel_b7,
    }
    if effnet_name not in builders:
        raise ValueError(f"Unknown effnet: {effnet_name}")
    return builders[effnet_name](num_classes=num_classes, concat_input=concat_input, pretrained=pretrained)


def main():
    opts = get_argparser().parse_args()

    random.seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    torch.manual_seed(opts.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opts.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[OK] device =", device, flush=True)

    exp_dir = os.path.join(opts.out_root, opts.exp_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    csv_path = os.path.join(exp_dir, "history.csv")

    C = opts.num_classes
    header_cols = [
        "epoch", "lr",
        "train_loss", "train_ce", "train_dice_loss",
        "train_macro_dice", "train_macro_jac", "train_macro_hd95",
    ]
    for c in range(C):
        header_cols += [f"train_dice_c{c}", f"train_jac_c{c}", f"train_hd95_c{c}"]
    header_cols += [
        "test_loss", "test_ce", "test_dice_loss",
        "test_macro_dice", "test_macro_jac", "test_macro_hd95",
    ]
    for c in range(C):
        header_cols += [f"test_dice_c{c}", f"test_jac_c{c}", f"test_hd95_c{c}"]
    header = ",".join(header_cols) + "\n"

    print("[OK] building datasets...", flush=True)
    tr_transform = RandomGenerator((opts.img_size, opts.img_size))
    train_ds = SynapseDataset(opts.root_dir, split="train", transform=tr_transform, strict=True, verbose=False)
    test_base = SynapseDataset(opts.root_dir, split="test_vol", transform=None, strict=True, verbose=False)
    test_ds = TestVol2DSliceDataset(test_base, transform=TestSliceTransform((opts.img_size, opts.img_size)))

    print(f"[OK] train={len(train_ds)} | test_vol={len(test_base)} | test_slices={len(test_ds)}", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=opts.batch_size, shuffle=True,
        num_workers=opts.num_workers, pin_memory=bool(opts.pin_memory),
        persistent_workers=(opts.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=opts.test_batch_size, shuffle=False,
        num_workers=opts.num_workers, pin_memory=bool(opts.pin_memory),
        persistent_workers=(opts.num_workers > 0),
    )

    print("[OK] building model (EfficientUNet)...", flush=True)
    model = _build_model(
        effnet_name=opts.effnet,
        num_classes=C,
        pretrained=bool(opts.encoder_pretrained),
        concat_input=bool(opts.concat_input),
    ).to(device)
    print("[OK] model ready", flush=True)

    # optional sanity check
    try:
        with torch.no_grad():
            x = torch.randn(2, 3, opts.img_size, opts.img_size, device=device)
            y = model(x)
        print(f"[OK] sanity forward: {tuple(y.shape)} (expected: (B,{C},H,W))", flush=True)
    except Exception as e:
        print(f"[WARN] sanity forward failed: {e}", flush=True)

    print("[OK] estimating CE weights...", flush=True)
    ce_w, _counts = estimate_ce_weights_from_loader(
        train_loader, C, device,
        max_batches=opts.ce_weight_batches,
        clamp_min=opts.ce_w_min,
        clamp_max=opts.ce_w_max,
    )
    print("[OK] CE weights ready", flush=True)

    ce_loss = nn.CrossEntropyLoss(weight=ce_w, reduction="mean")
    dice_loss = MultiClassDiceLoss(include_background=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.LR, weight_decay=opts.weight_decay)

    last_path = os.path.join(ckpt_dir, "model_last.pth")
    best_path = os.path.join(ckpt_dir, "model_best.pth")

    if opts.save_best_on == "macro_dice":
        best_score = -1e9
        es = EarlyStopping(patience=opts.es_patience, min_delta=opts.es_min_delta, mode="max")
    else:
        best_score = 1e9
        es = EarlyStopping(patience=opts.es_patience, min_delta=opts.es_min_delta, mode="min")

    print("[OK] starting training loop...", flush=True)

    for epoch in range(opts.NB_EPOCH):
        lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{opts.NB_EPOCH - 1} | lr={lr:.6g}", flush=True)

        model.train()
        running_train = init_running_metrics(C)

        loss_sum = ce_sum = dice_sum = 0.0
        n_batches = 0

        for i, samples in enumerate(tqdm(train_loader, desc="TRAIN", total=len(train_loader))):
            imgs, gts = _prep_2d_batch(samples["image"], samples["label"], device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)

            loss_ce = ce_loss(logits, gts)
            loss_d = dice_loss(logits, gts)
            loss = loss_ce + loss_d

            loss.backward()
            optimizer.step()

            loss_sum += float(loss.detach().cpu())
            ce_sum += float(loss_ce.detach().cpu())
            dice_sum += float(loss_d.detach().cpu())
            n_batches += 1

            if opts.metrics_every > 0 and (i % opts.metrics_every == 0):
                with torch.no_grad():
                    pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy()
                    gt = gts.cpu().numpy()
                    for b in range(pred.shape[0]):
                        running_train = update_running_metrics(running_train, pred[b], gt[b], C)

        train_loss = loss_sum / max(1, n_batches)
        train_ce = ce_sum / max(1, n_batches)
        train_dice_l = dice_sum / max(1, n_batches)

        tr_per_class, tr_macro = finalize_metrics(running_train, C)
        tr_macro_dice, tr_macro_jac, tr_macro_hd95 = tr_macro
        tr_dice_arr, tr_jac_arr, tr_hd95_arr = _per_class_to_arrays(tr_per_class, C)

        te_per_class, te_macro, te_losses = evaluate_on_test_2d(model, test_loader, C, device, ce_loss, dice_loss)
        te_macro_dice, te_macro_jac, te_macro_hd95 = te_macro
        test_loss, test_ce, test_dice_l = te_losses
        te_dice_arr, te_jac_arr, te_hd95_arr = _per_class_to_arrays(te_per_class, C)

        print(f"[TRAIN] loss={train_loss:.6f} ce={train_ce:.6f} dl={train_dice_l:.6f} | mdice={tr_macro_dice:.4f}", flush=True)
        print(f"[TEST ] loss={test_loss:.6f} ce={test_ce:.6f} dl={test_dice_l:.6f} | mdice={te_macro_dice:.4f}", flush=True)

        torch.save(model.state_dict(), last_path)

        row = []
        row += [str(epoch), str(lr)]
        row += [str(train_loss), str(train_ce), str(train_dice_l)]
        row += [_nan_to_empty(tr_macro_dice), _nan_to_empty(tr_macro_jac), _nan_to_empty(tr_macro_hd95)]
        for c in range(C):
            row += [_nan_to_empty(tr_dice_arr[c]), _nan_to_empty(tr_jac_arr[c]), _nan_to_empty(tr_hd95_arr[c])]
        row += [str(test_loss), str(test_ce), str(test_dice_l)]
        row += [_nan_to_empty(te_macro_dice), _nan_to_empty(te_macro_jac), _nan_to_empty(te_macro_hd95)]
        for c in range(C):
            row += [_nan_to_empty(te_dice_arr[c]), _nan_to_empty(te_jac_arr[c]), _nan_to_empty(te_hd95_arr[c])]
        append_csv(csv_path, header, ",".join(row) + "\n")

        if opts.save_best_on == "macro_dice":
            score = te_macro_dice
            if score is not None and not (isinstance(score, float) and np.isnan(score)) and float(score) > best_score:
                best_score = float(score)
                torch.save(model.state_dict(), best_path)
                print(f"[OK] best saved: macro_dice={best_score:.4f}", flush=True)
        else:
            score = train_loss
            if float(score) < best_score:
                best_score = float(score)
                torch.save(model.state_dict(), best_path)
                print(f"[OK] best saved: train_loss={best_score:.6f}", flush=True)

        if opts.early_stop == 1:
            monitor = te_macro_dice if opts.save_best_on == "macro_dice" else train_loss
            improved, should_stop = es.step(monitor)
            print(f"[ES] improved={improved} bad={es.bad_epochs}/{es.patience} best={es.best}", flush=True)
            if should_stop:
                print("[ES] stop", flush=True)
                break

    print("done", flush=True)


if __name__ == "__main__":
    main()