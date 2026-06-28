#!/usr/bin/env python3
import torch
from models.efficientvit_seg.dataset_brats import BraTSSliceDataset
if __name__ == '__main__':
    data_path = './data/002_BraTS21'
    print(f'🔍 Checking label values in: {data_path}')
    ds = BraTSSliceDataset(data_path, split='train', img_size=512)
    all_unique = set()
    bad_samples = []
    for i in range(len(ds)):
        _, lbl = ds[i]
        uniq = torch.unique(lbl).tolist()
        all_unique.update(uniq)
        if any((u < 0 or u > 3 for u in uniq)):
            bad_samples.append((i, uniq))
        if i < 5:
            print(f'🧩 Sample {i}: unique labels = {uniq}')
    print('\n📊 Overall unique label values in dataset:', sorted(all_unique))
    if bad_samples:
        print(f'❌ Found {len(bad_samples)} samples with out-of-range labels!')
        for i, uniq in bad_samples[:10]:
            print(f'   ⚠️ Sample {i}: {uniq}')
    else:
        print('✅ All label values are within [0, 3]. Dataset OK!')
