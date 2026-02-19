import sys
import os

# ✅ Make imports robust: add project root (folder of this script) to PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from networks_ibrahim import A1215_UNet_Resnet50_binary_base_pretrain_Multilabel as SegNet
from tqdm import tqdm
import utils
import random
import argparse
import numpy as np
from torch.utils.data import DataLoader
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
import torch
import torch.nn as nn
import datetime

# ✅ Multi-label metrics import (your MultiLabel_metric.py must be reachable from PROJECT_ROOT)
try:
    from MultiLabel_metric import multilabel_metric
except Exception as e:
    print(
        "[WARN] Could not import MultiLabel_metric.py. "
        "Make sure it's in the same folder as this script or in PYTHONPATH.\n"
        f"Error: {e}"
    )
    multilabel_metric = None

torch.cuda.set_device(0)


def make_print_to_file(path="./logs"):
    """
    Redirect all print() output to a dated log file in `path`,
    while still printing to the console.
    """
    import os
    import sys
    import datetime

    os.makedirs(path, exist_ok=True)

    class Logger(object):
        def __init__(self, filename="Default.log", path="."):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding="utf8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    fileName = datetime.datetime.now().strftime("DAY%Y_%m_%d_")
    sys.stdout = Logger(fileName + "Synapse_UNet_Resnet50Encoder.log", path=path)


make_print_to_file(path="./logs")


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Synapse", help="experiment_name")

    parser.add_argument(
        "--synapse_root",
        type=str,
        default=r"C:/Users/ia260111/OneDrive - DVHE/Bureau/Synapse_Baseline_allbackbones/Data/synapse",
        help="Synapse root folder containing train_npz/, test_vol_h5/, lists_Synapse/",
    )

    parser.add_argument("--num_classes", type=int, default=9, help="output channel of network")

    # ✅ reduced epochs (minimum change)
    parser.add_argument("--START_EPOCH", type=int, default=0)
    parser.add_argument("--NB_EPOCH", type=int, default=150)  # ✅ reduced from 230

    parser.add_argument("--LR", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)

    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID")
    parser.add_argument("--gpu_ids", type=list, default=[0], help="GPU ID(s)")

    parser.add_argument("--RESUME", type=bool, default=False)
    parser.add_argument("--patience", type=int, default=20)  # ✅ reduced (train-loss early stop only)

    parser.add_argument("--random_seed", type=int, default=1234)

    # ✅ metrics options
    parser.add_argument(
        "--log_metrics",
        action="store_true",
        default=True,
        help="Log multilabel metrics (Dice/Jaccard/HD95) from MultiLabel_metric.py",
    )
    parser.add_argument(
        "--metrics_every",
        type=int,
        default=10,  # ✅ safer default (HD95 is slow)
        help="Compute metrics every N batches (HD95 can be slow). 1 = every batch.",
    )

    return parser


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return 1 - loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), (
            f"predict {inputs.size()} & target {target.size()} shape do not match"
        )
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / self.n_classes


def init_running_metrics(num_classes: int):
    return {
        "dice_sum": {c: 0.0 for c in range(1, num_classes)},
        "jac_sum": {c: 0.0 for c in range(1, num_classes)},
        "hd95_sum": {c: 0.0 for c in range(1, num_classes)},
        "count": {c: 0 for c in range(1, num_classes)},
    }


def update_multilabel_running_stats(running, pred_2d, gt_2d, num_classes: int):
    if multilabel_metric is None:
        return running

    metric_list = multilabel_metric(pred_2d, gt_2d, num_classes)
    for cls_idx, (dice, jac, hd95) in enumerate(metric_list, start=1):
        running["dice_sum"][cls_idx] += float(dice)
        running["jac_sum"][cls_idx] += float(jac)
        running["hd95_sum"][cls_idx] += float(hd95)
        running["count"][cls_idx] += 1
    return running


def finalize_multilabel_stats(running, num_classes: int):
    per_class = {}
    dice_vals, jac_vals, hd95_vals = [], [], []

    for c in range(1, num_classes):
        cnt = running["count"][c]
        if cnt > 0:
            d = running["dice_sum"][c] / cnt
            j = running["jac_sum"][c] / cnt
            h = running["hd95_sum"][c] / cnt
        else:
            d, j, h = 0.0, 0.0, 0.0

        per_class[c] = (d, j, h)
        dice_vals.append(d)
        jac_vals.append(j)
        hd95_vals.append(h)

    macro = (
        float(np.mean(dice_vals)) if dice_vals else 0.0,
        float(np.mean(jac_vals)) if jac_vals else 0.0,
        float(np.mean(hd95_vals)) if hd95_vals else 0.0,
    )
    return per_class, macro


results_out_dir = "./Results_out/"
os.makedirs(results_out_dir, exist_ok=True)


def compute_lr(epoch: int, base_lr: float) -> float:
    """
    ✅ minimal, stable schedule:
    - warmup 5 epochs
    - constant until epoch 100
    - then decay
    """
    if epoch < 5:
        return base_lr * (epoch + 1) / 5.0
    if epoch < 100:
        return base_lr
    if epoch < 130:
        return base_lr / 2
    return base_lr / 4


def set_optimizer_lr(optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def main():
    opts = get_argparser().parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Synapse root:", opts.synapse_root)

    # Seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    tr_transform = RandomGenerator(output_size=[opts.img_size, opts.img_size])

    train_dataset = Synapse_dataset(
        root_dir=opts.synapse_root,
        split="train",
        transform=tr_transform,
        list_dir_name="lists_Synapse",
    )
    print("The length of train set is:", len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True)

    model_name = "./checkpoints/Synapse_UNet_Resnet50Encoder.pth"
    os.makedirs(os.path.dirname(model_name), exist_ok=True)

    # Model
    if len(opts.gpu_ids) > 1:
        model = SegNet.UNetWithResnet50Encoder(n_classes=opts.num_classes).to(device)
        if opts.RESUME and os.path.isfile(model_name):
            model.load_state_dict(torch.load(model_name, map_location=device))
        model = torch.nn.DataParallel(model)
    else:
        model = SegNet.UNetWithResnet50Encoder(n_classes=opts.num_classes).to(device)
        if opts.RESUME and os.path.isfile(model_name):
            model.load_state_dict(torch.load(model_name, map_location=device))

    # Loss
    criterion_seg = nn.CrossEntropyLoss(reduction="mean")
    criterion_dice = DiceLoss(n_classes=opts.num_classes)

    # ✅ CRITICAL FIX: create optimizer ONCE (do not recreate every epoch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.LR, weight_decay=opts.weight_decay)

    tm = datetime.datetime.now().strftime("T" + "%m%d%H%M")
    results_file_name = os.path.join(results_out_dir, tm + "Synapse_UNet_Resnet50Encoder_results.txt")

    best_train_loss = float("inf")
    early_stopping_counter = 0

    for epoch in range(opts.START_EPOCH, opts.NB_EPOCH):
        lr = compute_lr(epoch, opts.LR)
        set_optimizer_lr(optimizer, lr)

        print("current epoch:", epoch, "current lr:", lr)

        model.train()
        list_loss, list_loss_seg, list_loss_dice = [], [], []

        running_metrics = init_running_metrics(opts.num_classes)

        for i, samples in tqdm(enumerate(train_loader), total=len(train_loader)):
            images = samples["image"]
            labels = samples["label"]

            if images.size(1) == 1:
                images = images.repeat(1, 3, 1, 1)

            imgs_aug = images.to(device, dtype=torch.float32)
            lbs_aug = labels.to(device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)

            out_seg = model(imgs_aug)  # (B,C,H,W)

            loss_seg = criterion_seg(out_seg, lbs_aug)
            loss_dice = criterion_dice(out_seg, lbs_aug)
            loss = loss_seg + loss_dice

            loss.backward()
            optimizer.step()

            list_loss.append(loss.detach())
            list_loss_seg.append(loss_seg.detach())
            list_loss_dice.append(loss_dice.detach())

            # ✅ metrics (every N batches)
            if (
                opts.log_metrics
                and multilabel_metric is not None
                and opts.metrics_every > 0
                and (i % opts.metrics_every == 0)
            ):
                with torch.no_grad():
                    pred = torch.argmax(torch.softmax(out_seg, dim=1), dim=1)  # (B,H,W)
                    for b in range(pred.shape[0]):
                        pred_2d = pred[b].detach().cpu().numpy()
                        gt_2d = lbs_aug[b].detach().cpu().numpy()
                        running_metrics = update_multilabel_running_stats(
                            running_metrics, pred_2d, gt_2d, opts.num_classes
                        )

        # epoch losses
        epoch_loss = torch.stack(list_loss).mean() if list_loss else torch.tensor(0.0)
        epoch_loss_seg = torch.stack(list_loss_seg).mean() if list_loss_seg else torch.tensor(0.0)
        epoch_loss_dice = torch.stack(list_loss_dice).mean() if list_loss_dice else torch.tensor(0.0)

        # ✅ Save last checkpoint each epoch
        torch.save(model.state_dict(), model_name)

        # early stop on train loss (as requested)
        if epoch_loss.item() < best_train_loss:
            best_train_loss = epoch_loss.item()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= opts.patience:
            print(f"Early stopping after {epoch + 1} epochs.")
            break

        # finalize metrics
        per_class, macro = finalize_multilabel_stats(running_metrics, opts.num_classes)
        macro_dice, macro_jac, macro_hd95 = macro

        print(
            "Epoch %d, Loss=%f, Loss_SEG=%f, Loss_DICE=%f"
            % (epoch, epoch_loss.item(), epoch_loss_seg.item(), epoch_loss_dice.item())
        )

        if opts.log_metrics and multilabel_metric is not None:
            print(
                f"[METRICS] Macro (classes 1..{opts.num_classes-1}) "
                f"Dice={macro_dice:.4f} | Jaccard={macro_jac:.4f} | HD95={macro_hd95:.4f}"
            )
            for c in range(1, opts.num_classes):
                d, j, h = per_class[c]
                print(f"[METRICS] Class {c:02d}: Dice={d:.4f} | Jaccard={j:.4f} | HD95={h:.4f}")

        with open(results_file_name, "a", encoding="utf8") as file:
            file.write(
                "Epoch %d, Loss=%f, Loss_SEG=%f, Loss_DICE=%f\n"
                % (epoch, epoch_loss.item(), epoch_loss_seg.item(), epoch_loss_dice.item())
            )
            if opts.log_metrics and multilabel_metric is not None:
                file.write(
                    "[METRICS] Macro Dice=%.4f | Jaccard=%.4f | HD95=%.4f\n"
                    % (macro_dice, macro_jac, macro_hd95)
                )
                for c in range(1, opts.num_classes):
                    d, j, h = per_class[c]
                    file.write(
                        "[METRICS] Class %02d: Dice=%.4f | Jaccard=%.4f | HD95=%.4f\n"
                        % (c, d, j, h)
                    )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
