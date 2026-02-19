import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom  # ✅ no deprecation warning
from torch.utils.data import Dataset
from pathlib import Path


# -------------------------
# Augmentations
# -------------------------
def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size  # (H, W)

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        x, y = image.shape
        if (x != self.output_size[0]) or (y != self.output_size[1]):
            # image: cubic interpolation
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            # label: nearest neighbor to preserve class ids
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # (1, H, W)
        label = torch.from_numpy(label.astype(np.int64))                 # (H, W)
        return {"image": image, "label": label}


# -------------------------
# Dataset
# -------------------------
class Synapse_dataset(Dataset):
    """
    Folder architecture expected:

    root_dir/
      train_npz/        (for train split) -> caseXXXX_sliceYYY.npz
      test_vol_h5/      (for val/test split) -> caseXXXX.npy.h5
      lists_Synapse/    -> train.txt, test_vol.txt, all.lst
    """

    def __init__(self, root_dir, split="train", transform=None, list_dir_name="lists_Synapse"):
        self.transform = transform
        self.split = split.lower().strip()

        self.root_dir = Path(root_dir).expanduser().resolve()
        self.list_dir = (self.root_dir / list_dir_name).resolve()

        # where the actual data lives
        self.train_npz_dir = (self.root_dir / "train_npz").resolve()
        self.test_h5_dir = (self.root_dir / "test_vol_h5").resolve()

        # choose list file depending on split
        if self.split == "train":
            list_file = self.list_dir / "train.txt"
        else:
            # many repos use test_vol.txt for evaluation volumes
            list_file = self.list_dir / "test_vol.txt"

        print("ROOT DIR     :", self.root_dir)
        print("LIST DIR     :", self.list_dir)
        print("LIST FILE    :", list_file)
        print("TRAIN NPZ DIR:", self.train_npz_dir)
        print("TEST  H5 DIR :", self.test_h5_dir)

        if not list_file.is_file():
            raise FileNotFoundError(f"Missing list file: {list_file}")

        self.sample_list = list_file.read_text().splitlines()

        # basic sanity checks (not mandatory but helpful)
        if self.split == "train" and not self.train_npz_dir.is_dir():
            raise FileNotFoundError(f"Missing folder: {self.train_npz_dir}")
        if self.split != "train" and not self.test_h5_dir.is_dir():
            raise FileNotFoundError(f"Missing folder: {self.test_h5_dir}")

    def __len__(self):
        return len(self.sample_list)

    def show_paths(self, max_items=20):
        n = min(len(self.sample_list), max_items)
        print(f"Showing {n}/{len(self.sample_list)} paths for split='{self.split}'")

        for i in range(n):
            name = self.sample_list[i].strip()
            if self.split == "train":
                p = self.train_npz_dir / f"{name}.npz"
            else:
                p = self.test_h5_dir / f"{name}.npy.h5"
            print(p, "| exists =", p.exists())

    def __getitem__(self, idx):
        name = self.sample_list[idx].strip()

        if self.split == "train":
            data_path = self.train_npz_dir / f"{name}.npz"
            if not data_path.is_file():
                raise FileNotFoundError(f"Missing npz: {data_path}")

            data = np.load(str(data_path))
            image, label = data["image"], data["label"]

        else:
            # volumes for evaluation
            h5_path = self.test_h5_dir / f"{name}.npy.h5"
            if not h5_path.is_file():
                raise FileNotFoundError(f"Missing h5: {h5_path}")

            with h5py.File(str(h5_path), "r") as data:
                image, label = data["image"][:], data["label"][:]

        sample = {"image": image, "label": label}

        if self.transform is not None:
            sample = self.transform(sample)

        sample["case_name"] = name
        return sample


# -------------------------
# Quick local test
# -------------------------
if __name__ == "__main__":
    root_dir = r"C:/Users/ia260111/OneDrive - DVHE/Bureau/Synapse_Baseline_allbackbones/Data/synapse"

    ds_train = Synapse_dataset(root_dir=root_dir, split="train", transform=None)
    ds_train.show_paths(max_items=30)

    ds_test = Synapse_dataset(root_dir=root_dir, split="test", transform=None)
    ds_test.show_paths(max_items=10)
