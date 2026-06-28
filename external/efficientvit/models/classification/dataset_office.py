import os
from PIL import Image
from torch.utils.data import Dataset


class OfficeDataset(Dataset):

    def __init__(self, txt_file, transform=None, class_to_idx=None, return_path=False):

        self.samples = []
        self.transform = transform
        self.return_path = return_path

        with open(txt_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            path, label = line.strip().rsplit(" ", 1)
            self.samples.append((path, label))

        # use shared mapping
        if class_to_idx is None:
            labels = sorted(list(set([x[1] for x in self.samples])))
            self.class_to_idx = {l: i for i, l in enumerate(labels)}
        else:
            self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_path, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.class_to_idx[label]

        # --------------------------------
        # training mode
        # --------------------------------
        if not self.return_path:
            return img, label

        # --------------------------------
        # ORIENT extraction mode
        # --------------------------------
        return img, label, img_path