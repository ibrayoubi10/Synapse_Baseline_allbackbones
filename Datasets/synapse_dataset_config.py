# author: Ibrahim (refactor)
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from pathlib import Path

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label, angle_range=(-20, 20)):
    angle = np.random.randint(angle_range[0], angle_range[1])
    image = ndimage.rotate(image, angle, order=3, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size):
        # output_size: (H, W)
        self.output_size = tuple(output_size)

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if (x, y) != self.output_size:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # normalize image to float32 
        image = torch.from_numpy(image.astype("float32")).unsqueeze(0)  # (1,H,W)
        label = torch.from_numpy(label.astype("int64"))               # (H,W)
        return {"image": image, "label": label}

class SynapseDataset(Dataset):
    """
    Usage:
      ds = SynapseDataset(root_dir="/path/Synapse", split="train", transform=RandomGenerator((512,512)))
    Expects:
      root_dir/
        train_npz/
        test_vol_h5/
        lists_Synapse/
    """
    def __init__(self,
                 root_dir,
                 split="train",
                 transform=None,
                 list_dir_name="lists_Synapse",
                 train_npz_subdir="train_npz",
                 test_h5_subdir="test_vol_h5",
                 train_ext=".npz",
                 test_ext=".npy.h5",
                 strict=True,
                 verbose=False):
        self.transform = transform
        self.split = str(split)
        self.root = Path(root_dir).expanduser().resolve()
        self.list_dir = (self.root / list_dir_name).resolve()
        self.train_dir = (self.root / train_npz_subdir).resolve()
        self.test_dir = (self.root / test_h5_subdir).resolve()
        self.train_ext = str(train_ext)
        self.test_ext = str(test_ext)
        self.strict = bool(strict)
        self.verbose = bool(verbose)
        list_file = f"{self.split}.txt"
        self.list_path = (self.list_dir / list_file).resolve()
        if self.verbose:
            print("[SynapseDataset] root:", self.root)
            print("[SynapseDataset] list:", self.list_path)
            print("[SynapseDataset] train_dir:", self.train_dir)
            print("[SynapseDataset] test_dir:", self.test_dir)
        if self.strict:
            if not self.root.is_dir():
                raise FileNotFoundError(self.root)
            if not self.list_dir.is_dir():
                raise FileNotFoundError(self.list_dir)
            if not self.list_path.is_file():
                raise FileNotFoundError(self.list_path)
        lines = self.list_path.read_text().splitlines() if self.list_path.is_file() else []
        self.sample_list = [x.strip() for x in lines if x.strip()]
        if self.strict and len(self.sample_list) == 0:
            raise ValueError(f"Empty list: {self.list_path}")

    def __len__(self):
        return len(self.sample_list)

    def _path_for(self, name: str):
        name = name.strip()
        if self.split.lower() == "train":
            return self.train_dir / f"{name}{self.train_ext}"
        return self.test_dir / f"{name}{self.test_ext}"

    def __getitem__(self, idx):
        name = self.sample_list[idx]
        p = self._path_for(name)
        if self.strict and not p.is_file():
            raise FileNotFoundError(p)
        if self.split.lower() == "train":
            data = np.load(str(p))
            image, label = data["image"], data["label"]
        else:
            with h5py.File(str(p), "r") as f:
                image, label = f["image"][:], f["label"][:]
        sample = {"image": image, "label": label}
        if self.transform is not None:
            sample = self.transform(sample)
        sample["case_name"] = name
        return sample

if __name__ == "__main__":
    ds = SynapseDataset("/data3/nkozah/my_project/Data/synapse", split="train", transform=RandomGenerator((512,512)), verbose=True)
    print(len(ds))
    s = ds[0]
    print(s["image"].shape, s["label"].shape)