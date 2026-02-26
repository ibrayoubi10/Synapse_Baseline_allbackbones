# main.py 
import os
import sys
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../my_project/Ibrahim
PROJECT_ROOT = os.path.dirname(THIS_DIR)                      # .../my_project
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scipy.ndimage import zoom

from Ibrahim.Datasets.synapse_dataset_config import SynapseDataset, RandomGenerator
from Ibrahim.networks.Unet_Multilabel import U_Net
from Ibrahim.losses.DiceLoss import MultiClassDiceLoss
from Ibrahim.metrics.Multilabel_metrics import init_running_metrics, update_running_metrics, finalize_metrics


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


def estimate_dice_weights_from_ce_weights(ce_w):
    w = ce_w.clone().detach().float()
    w[0] = min(float(w[0]), 1.0)
    w = w / w.mean().clamp_min(1e-12)
    return w


def append_csv(csv_path, header, row):
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", encoding="utf-8") as f:
        if not exists:
            f.write(header)
        f.write(row)


def _prep_2d_batch(images, labels, img_size, device):
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


class TestSliceTransform:
    def __init__(self, output_size):
        self.output_size = tuple(output_size)

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        x, y = image.shape
        if (x, y) != self.output_size:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # (1,H,W)
        label = torch.from_numpy(label.astype(np.int64))                 # (H,W)
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
        sample = {
            "image": img[z],
            "label": lab[z],
            "case_name": s.get("case_name", str(case_idx)),
            "slice_idx": z,
        }
        if self.transform is not None:
            t = self.transform({"image": sample["image"], "label": sample["label"]})
            sample["image"], sample["label"] = t["image"], t["label"]
        return sample


def evaluate_on_test_2d(model, test_loader, num_classes, img_size, device, ce_loss, dice_loss):
    model.eval()
    running = init_running_metrics(num_classes)

    loss_sum = 0.0
    ce_sum = 0.0
    dice_sum = 0.0
    n_batches = 0

    with torch.no_grad():
        for samples in tqdm(test_loader, total=len(test_loader), desc="TEST_2D"):
            images = samples["image"]
            labels = samples["label"]
            imgs, gts = _prep_2d_batch(images, labels, img_size, device)

            logits = model(imgs)
            loss_ce = ce_loss(logits, gts)
            loss_d = dice_loss(logits, gts)
            loss = loss_ce + loss_d

            loss_sum += float(loss.detach().cpu())
            ce_sum += float(loss_ce.detach().cpu())
            dice_sum += float(loss_d.detach().cpu())
            n_batches += 1

            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            pred_np = pred.detach().cpu().numpy()
            gt_np = gts.detach().cpu().numpy()

            for b in range(pred_np.shape[0]):
                running = update_running_metrics(running, pred_np[b], gt_np[b], num_classes)

    per_class, macro = finalize_metrics(running, num_classes)
    test_loss = loss_sum / max(1, n_batches)
    test_ce = ce_sum / max(1, n_batches)
    test_dice_l = dice_sum / max(1, n_batches)
    return per_class, macro, (test_loss, test_ce, test_dice_l)


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
    p.add_argument("--exp_name", type=str, default="Synapse_UNet_2D")
    p.add_argument("--metrics_every", type=int, default=20)
    p.add_argument("--save_best_on", type=str, default="macro_dice", choices=["macro_dice", "loss"])
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
    csv_path = os.path.join(exp_dir, "history.csv")

    C = opts.num_classes

    header_cols = [
        "epoch", "lr",
        "train_loss", "train_ce", "train_dice_loss",
        "train_macro_dice", "train_macro_jac", "train_macro_hd95",
        "test_loss", "test_ce", "test_dice_loss",
        "test_macro_dice", "test_macro_jac", "test_macro_hd95",
    ]
    for c in range(1, C):
        header_cols += [
            f"train_dice_c{c:02d}", f"train_jac_c{c:02d}", f"train_hd95_c{c:02d}",
            f"test_dice_c{c:02d}",  f"test_jac_c{c:02d}",  f"test_hd95_c{c:02d}",
        ]
    header = ",".join(header_cols) + "\n"

    tr_transform = RandomGenerator((opts.img_size, opts.img_size))
    train_ds = SynapseDataset(opts.root_dir, split="train", transform=tr_transform, strict=True, verbose=False)
    train_loader = DataLoader(train_ds, batch_size=opts.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    test_base = SynapseDataset(opts.root_dir, split="test_vol", transform=None, strict=True, verbose=False)
    test_ds = TestVol2DSliceDataset(test_base, transform=TestSliceTransform((opts.img_size, opts.img_size)))
    test_loader = DataLoader(test_ds, batch_size=opts.test_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print("Train samples:", len(train_ds))
    print("Test volumes :", len(test_base))
    print("Test slices  :", len(test_ds))

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

    best_score = -1e9 if opts.save_best_on == "macro_dice" else 1e9
    last_path = os.path.join(ckpt_dir, "model_last.pth")
    best_path = os.path.join(ckpt_dir, "model_best.pth")

    def _fmt(x):
        return "" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x}"

    for epoch in range(opts.NB_EPOCH):
        lr = optimizer.param_groups[0]["lr"]

        print("\n" + "=" * 72)
        print(f"Epoch {epoch}/{opts.NB_EPOCH - 1} | lr={lr:.6g}")
        print("=" * 72)

        model.train()
        running_train = init_running_metrics(C)
        loss_sum = 0.0
        ce_sum = 0.0
        dice_sum = 0.0
        n_batches = 0

        for i, samples in enumerate(tqdm(train_loader, desc="TRAIN_2D", total=len(train_loader))):
            images = samples["image"]
            labels = samples["label"]
            imgs, gts = _prep_2d_batch(images, labels, opts.img_size, device)

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
                    pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                    pred_np = pred.detach().cpu().numpy()
                    gt_np = gts.detach().cpu().numpy()
                    for b in range(pred_np.shape[0]):
                        running_train = update_running_metrics(running_train, pred_np[b], gt_np[b], C)

        train_loss = loss_sum / max(1, n_batches)
        train_ce = ce_sum / max(1, n_batches)
        train_dice_l = dice_sum / max(1, n_batches)

        train_per_class, train_macro = finalize_metrics(running_train, C)
        tr_macro_dice, tr_macro_jac, tr_macro_hd95 = train_macro

        print("\n--- Train Summary ---")
        print(f"[TRAIN] Loss={train_loss:.6f} | CE={train_ce:.6f} | DiceLoss={train_dice_l:.6f}")
        print(f"[TRAIN] Macro Dice={tr_macro_dice:.4f} | Macro Jac={tr_macro_jac:.4f} | Macro HD95={tr_macro_hd95:.4f}")

        test_per_class, test_macro, test_losses = evaluate_on_test_2d(
            model, test_loader, C, opts.img_size, device, ce_loss, dice_loss
        )
        te_macro_dice, te_macro_jac, te_macro_hd95 = test_macro
        test_loss, test_ce, test_dice_l = test_losses

        print("\n--- Test Summary (2D slices from volumes) ---")
        print(f"[TEST]  Loss={test_loss:.6f} | CE={test_ce:.6f} | DiceLoss={test_dice_l:.6f}")
        print(f"[TEST]  Macro Dice={te_macro_dice:.4f} | Macro Jac={te_macro_jac:.4f} | Macro HD95={te_macro_hd95:.4f}")

        torch.save(model.state_dict(), last_path)
        print("Saved last checkpoint:", last_path)

        row = [
            str(epoch), f"{lr}",
            f"{train_loss}", f"{train_ce}", f"{train_dice_l}",
            _fmt(tr_macro_dice), _fmt(tr_macro_jac), _fmt(tr_macro_hd95),
            f"{test_loss}", f"{test_ce}", f"{test_dice_l}",
            _fmt(te_macro_dice), _fmt(te_macro_jac), _fmt(te_macro_hd95),
        ]
        for c in range(1, C):
            td, tj, th = train_per_class.get(c, (np.nan, np.nan, np.nan))
            vd, vj, vh = test_per_class.get(c, (np.nan, np.nan, np.nan))
            row += [_fmt(td), _fmt(tj), _fmt(th), _fmt(vd), _fmt(vj), _fmt(vh)]
        append_csv(csv_path, header, ",".join(row) + "\n")

        if opts.save_best_on == "macro_dice":
            score = te_macro_dice
            if (score is not None) and (not np.isnan(score)) and score > best_score:
                best_score = score
                torch.save(model.state_dict(), best_path)
                print(f"Saved best (macro_dice={best_score:.4f}) -> {best_path}")
        else:
            score = train_loss
            if score < best_score:
                best_score = score
                torch.save(model.state_dict(), best_path)
                print(f"Saved best (train_loss={best_score:.6f}) -> {best_path}")

    print("Training finished.")

if __name__ == "__main__":
    main()