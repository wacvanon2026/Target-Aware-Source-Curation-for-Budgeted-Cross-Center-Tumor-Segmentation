import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class BraTSSliceDataset(Dataset):

    def __init__(self, root_dir, split='train', img_size=512, split_txt_dir=None, skip_empty=True):
        assert split in ['train', 'val', 'test'], 'split must be one of train/val/test'
        self.img_size = img_size
        self.skip_empty = skip_empty
        self.return_meta = True
        selected_subjects = None
        if split_txt_dir is not None:
            split_file = os.path.join(split_txt_dir, f'{split}_subjects.txt')
            if os.path.exists(split_file):
                with open(split_file, 'r') as f:
                    selected_subjects = [line.strip() for line in f if line.strip()]
                print(f'✅ Loaded {len(selected_subjects)} subjects from {split_file}')
            else:
                print(f'⚠️ Split file not found: {split_file}, loading all subjects.')
        else:
            print('⚠️ No split_txt_dir provided — loading all subjects.')
        self.image_dir = os.path.join(root_dir, 'imagesTr')
        self.label_dir = os.path.join(root_dir, 'labelsTr')
        image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.npy')])
        label_files = sorted([f for f in os.listdir(self.label_dir) if f.endswith('.npy')])
        self.samples = []
        bg_skipped, non_selected = (0, 0)
        for img_f, lbl_f in zip(image_files, label_files):
            slice_name = img_f.replace('.npy', '')
            subject_id = slice_name.split('_slice')[0]
            slice_idx = int(slice_name.split('_slice')[1])
            if selected_subjects is not None:
                if slice_name not in selected_subjects and subject_id not in selected_subjects:
                    non_selected += 1
                    continue
            lbl_path = os.path.join(self.label_dir, lbl_f)
            lbl = np.load(lbl_path)
            if self.skip_empty and (not (lbl > 0).any()):
                bg_skipped += 1
                continue
            self.samples.append((img_f, lbl_f, subject_id, slice_idx))
        print(f'📊 Loaded {len(self.samples)} slices from {root_dir} (skipped {bg_skipped} background-only, {non_selected} non-selected)')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_f, lbl_f, subject_id, slice_idx = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_f)
        lbl_path = os.path.join(self.label_dir, lbl_f)
        img = np.load(img_path).astype(np.float32)
        lbl = np.load(lbl_path).astype(np.int64)
        if img.ndim == 2:
            img = img[:, :, None]
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        img = (img - img.min()) / (img.max() - img.min() + 1e-08)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        lbl = torch.from_numpy(lbl).long()
        img = F.interpolate(img.unsqueeze(0), size=(self.img_size, self.img_size), mode='bilinear', align_corners=False).squeeze(0)
        lbl = F.interpolate(lbl.unsqueeze(0).unsqueeze(0).float(), size=(self.img_size, self.img_size), mode='nearest').squeeze().long()
        if self.return_meta:
            return (img, lbl, subject_id, slice_idx)
        else:
            return (img, lbl)
