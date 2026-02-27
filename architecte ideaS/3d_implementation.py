# main.py 
import os
import sys
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from scipy.ndimage import zoom

from medpy import metric as medpy_metric

THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../my_project/Ibrahim
PROJECT_ROOT = os.path.dirname(THIS_DIR)                      # .../my_project
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Ibrahim.Datasets.synapse_dataset_config import SynapseDataset, RandomGenerator
from Ibrahim.networks.Unet_Multilabel import U_Net
from Ibrahim.losses.DiceLoss import MultiClassDiceLoss


def estimate_ce_weights_from_loader(train_loader, num_classes, device, max_batches=200, clamp_min=0.1, clamp_max=10.0):
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for k, s in enumerate(train_loader):
        if k >= max_batches:
            break
        y = s["label"].view(-1)
        binc = torch.bincount(y, minlength=num_classes).double()
        counts += binc
    freq = counts / counts.sum().clamp_min(1.0)
    w = 1.0 / (freq + 1e-12)
    w = w / w.mean().clamp_min(1e-12)
    w = w.float()
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
    if images.size(1) == 1:
        images = images.repeat(1, 3, 1, 1)
    imgs = images.to(device, dtype=torch.float32, non_blocking=True)
    gts = labels.to(device, dtype=torch.long, non_blocking=True)
    return imgs, gts


def _ensure_depth_first(vol_np):
    if vol_np.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {vol_np.shape}")
    depth_axis = int(np.argmin(list(vol_np.shape)))
    if depth_axis != 0:
        vol_np = np.moveaxis(vol_np, depth_axis, 0)
    return vol_np  # (D,H,W)


def _resize_slice_to(img2d, lab2d, out_hw):
    Ht, Wt = out_hw
    H, W = img2d.shape
    if (H, W) == (Ht, Wt):
        return img2d, lab2d
    img_r = zoom(img2d, (Ht / H, Wt / W), order=3)
    lab_r = zoom(lab2d, (Ht / H, Wt / W), order=0)
    return img_r, lab_r


def _resize_pred_back(pred2d, out_hw):
    Ht, Wt = out_hw
    H, W = pred2d.shape
    if (H, W) == (Ht, Wt):
        return pred2d
    return zoom(pred2d, (Ht / H, Wt / W), order=0)


def _metric_per_class_3d(pred_vol, gt_vol, c):
    pred = (pred_vol == c).astype(np.uint8)
    gt = (gt_vol == c).astype(np.uint8)

    if gt.sum() == 0:
        return np.nan, np.nan, np.nan
    if pred.sum() == 0:
        return 0.0, 0.0, np.nan

    dice = medpy_metric.binary.dc(pred, gt)
    jac = medpy_metric.binary.jc(pred, gt)
    hd95 = medpy_metric.binary.hd95(pred, gt)
    return float(dice), float(jac), float(hd95)


def evaluate_on_test_3d(model, test_loader, num_classes, img_size, device, ce_loss, dice_loss):
    model.eval()

    # losses averaged over all processed slices (like your 2D test)
    loss_sum = 0.0
    ce_sum = 0.0
    dice_sum = 0.0
    n_slices = 0

    # per-class metrics averaged over volumes (nan-safe)
    sums = {c: np.array([0.0, 0.0, 0.0], dtype=np.float64) for c in range(1, num_classes)}
    counts = {c: np.array([0, 0, 0], dtype=np.int64) for c in range(1, num_classes)}  # valid counts per metric

    with torch.no_grad():
        for samples in tqdm(test_loader, total=len(test_loader), desc="TEST_3D"):
            # batch_size=1 expected
            img = samples["image"]
            lab = samples["label"]

            # to numpy volume
            if torch.is_tensor(img):
                img_np = img.squeeze(0).cpu().numpy()
            else:
                img_np = np.asarray(img)[0]
            if torch.is_tensor(lab):
                lab_np = lab.squeeze(0).cpu().numpy()
            else:
                lab_np = np.asarray(lab)[0]

            img_np = _ensure_depth_first(img_np)
            lab_np = _ensure_depth_first(lab_np)

            D, H0, W0 = img_np.shape
            pred_vol = np.zeros((D, H0, W0), dtype=np.int64)

            for z in range(D):
                slice_img = img_np[z]
                slice_lab = lab_np[z]

                # resize to network input
                slice_img_r, slice_lab_r = _resize_slice_to(slice_img, slice_lab, (img_size, img_size))

                # torch tensors
                x = torch.from_numpy(slice_img_r.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
                y = torch.from_numpy(slice_lab_r.astype(np.int64)).unsqueeze(0)                 # (1,H,W)
                x, y = _prep_2d_batch(x, y, device)

                logits = model(x)
                l_ce = ce_loss(logits, y)
                l_d = dice_loss(logits, y)
                l = l_ce + l_d

                loss_sum += float(l.detach().cpu())
                ce_sum += float(l_ce.detach().cpu())
                dice_sum += float(l_d.detach().cpu())
                n_slices += 1

                p = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0).detach().cpu().numpy().astype(np.int64)

                # resize prediction back to original slice size
                p_back = _resize_pred_back(p, (H0, W0))
                pred_vol[z] = p_back

            # per-class 3D metrics on whole volume
            for c in range(1, num_classes):
                d, j, h = _metric_per_class_3d(pred_vol, lab_np, c)

                if not np.isnan(d):
                    sums[c][0] += d
                    counts[c][0] += 1
                if not np.isnan(j):
                    sums[c][1] += j
                    counts[c][1] += 1
                if not np.isnan(h):
                    sums[c][2] += h
                    counts[c][2] += 1

    test_loss = loss_sum / max(1, n_slices)
    test_ce = ce_sum / max(1, n_slices)
    test_dice_l = dice_sum / max(1, n_slices)

    test_per_class = {}
    for c in range(1, num_classes):
        d = sums[c][0] / counts[c][0] if counts[c][0] > 0 else np.nan
        j = sums[c][1] / counts[c][1] if counts[c][1] > 0 else np.nan
        h = sums[c][2] / counts[c][2] if counts[c][2] > 0 else np.nan
        test_per_class[c] = (float(d) if not np.isnan(d) else np.nan,
                             float(j) if not np.isnan(j) else np.nan,
                             float(h) if not np.isnan(h) else np.nan)

    return test_per_class, (test_loss, test_ce, test_dice_l)


def get_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", type=str, default="/data3/nkozah/my_project/Data/synapse")
    p.add_argument("--num_classes", type=int, default=9)
    p.add_argument("--NB_EPOCH", type=int, default=200)
    p.add_argument("--LR", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--exp_name", type=str, default="Synapse_UNet")
    p.add_argument("--random_seed", type=int, default=1234)
    p.add_argument("--ce_weight_batches", type=int, default=200)
    p.add_argument("--ce_w_min", type=float, default=0.1)
    p.add_argument("--ce_w_max", type=float, default=10.0)
    p.add_argument("--out_root", type=str, default="/data3/nkozah/my_project/Ibrahim/experiments")
    return p


def main():
    opts = get_argparser().parse_args()
    random.seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    torch.manual_seed(opts.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    exp_dir = os.path.join(opts.out_root, opts.exp_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    csv_path = os.path.join(exp_dir, "history_UNet.csv")

    C = opts.num_classes

    # CSV header: ONLY train/test losses + test per-class metrics
    header_cols = [
        "epoch", "lr",
        "train_loss", "train_ce", "train_dice_loss",
        "test_loss", "test_ce", "test_dice_loss",
    ]
    for c in range(1, C):
        header_cols += [f"test_dice_c{c:02d}", f"test_jac_c{c:02d}", f"test_hd95_c{c:02d}"]
    header = ",".join(header_cols) + "\n"

    tr_transform = RandomGenerator((opts.img_size, opts.img_size))
    train_ds = SynapseDataset(opts.root_dir, split="train", transform=tr_transform, strict=True, verbose=False)
    train_loader = DataLoader(train_ds, batch_size=opts.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # 3D test loader: volumes directly
    test_ds = SynapseDataset(opts.root_dir, split="test_vol", transform=None, strict=True, verbose=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    print("Train samples:", len(train_ds))
    print("Test volumes :", len(test_ds))

    model = U_Net(img_ch=3, num_class=C).to(device)

    ce_w, counts = estimate_ce_weights_from_loader(
        train_loader, C, device,
        max_batches=opts.ce_weight_batches,
        clamp_min=opts.ce_w_min,
        clamp_max=opts.ce_w_max,
    )
    print("Pixel counts:", counts.cpu().numpy().astype(np.int64))
    print("CE weights:", ce_w.detach().cpu().numpy())

    ce_loss = nn.CrossEntropyLoss(weight=ce_w, reduction="mean")
    dice_loss = MultiClassDiceLoss(include_background=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.LR, weight_decay=opts.weight_decay)

    last_path = os.path.join(ckpt_dir, "model_last.pth")

    def _fmt(x):
        return "" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x}"

    for epoch in range(opts.NB_EPOCH):
        lr = optimizer.param_groups[0]["lr"]

        print("\n" + "=" * 72)
        print(f"Epoch {epoch}/{opts.NB_EPOCH - 1} | lr={lr:.6g}")
        print("=" * 72)

        # -------------------
        # TRAIN (losses only)
        # -------------------
        model.train()
        loss_sum = 0.0
        ce_sum = 0.0
        dice_sum = 0.0
        n_batches = 0

        for samples in tqdm(train_loader, desc="TRAIN_2D", total=len(train_loader)):
            images = samples["image"]
            labels = samples["label"]
            imgs, gts = _prep_2d_batch(images, labels, device)

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

        train_loss = loss_sum / max(1, n_batches)
        train_ce = ce_sum / max(1, n_batches)
        train_dice_l = dice_sum / max(1, n_batches)

        print(f"[TRAIN] Loss={train_loss:.6f} | CE={train_ce:.6f} | DiceLoss={train_dice_l:.6f}")

        # -------------------
        # TEST (3D volume)
        # -------------------
        test_per_class, test_losses = evaluate_on_test_3d(
            model, test_loader, C, opts.img_size, device, ce_loss, dice_loss
        )
        test_loss, test_ce, test_dice_l = test_losses

        print(f"[TEST_3D] Loss={test_loss:.6f} | CE={test_ce:.6f} | DiceLoss={test_dice_l:.6f}")
        for c in range(1, C):
            d, j, h = test_per_class.get(c, (np.nan, np.nan, np.nan))
            print(f"[TEST_3D] Class {c:02d}: Dice={_fmt(d)} | Jac={_fmt(j)} | HD95={_fmt(h)}")

        # checkpoint
        torch.save(model.state_dict(), last_path)
        print("Saved last checkpoint:", last_path)

        # CSV row
        row = [
            str(epoch), f"{lr}",
            f"{train_loss}", f"{train_ce}", f"{train_dice_l}",
            f"{test_loss}", f"{test_ce}", f"{test_dice_l}",
        ]
        for c in range(1, C):
            d, j, h = test_per_class.get(c, (np.nan, np.nan, np.nan))
            row += [_fmt(d), _fmt(j), _fmt(h)]
        append_csv(csv_path, header, ",".join(row) + "\n")

    print("Training finished.")


if __name__ == "__main__":
    main()