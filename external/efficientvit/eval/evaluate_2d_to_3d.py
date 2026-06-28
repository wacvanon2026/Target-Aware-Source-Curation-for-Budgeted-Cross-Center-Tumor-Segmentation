#!/usr/bin/env python3
import os, sys, yaml
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.efficientvit_seg.efficientvit_seg import EfficientViT_Seg
from models.efficientvit_seg.dataset_brats import BraTSSliceDataset


# -----------------------
# Dice for 3D volumes
# -----------------------
def dice_3d(pred, gt):
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    inter = (pred & gt).sum()
    return 2 * inter / (pred.sum() + gt.sum() + 1e-8)


# -----------------------
# Main evaluation (OOM-safe)
# -----------------------
def evaluate_2d_to_3d(config_path, checkpoint_path):
    print("📄 Loading config:", config_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 Using device:", device)

    # Load model
    model_cfg = cfg["model"]
    model = EfficientViT_Seg(
        backbone=model_cfg["name"],
        in_channels=model_cfg["in_channels"],
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg.get("pretrained", False)
    ).to(device)

    print("📦 Loading checkpoint:", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    data_cfg = cfg["data"]["test"]
    img_size = cfg["data"]["img_size"]

    # Load dataset
    print("📂 Preparing test dataset...")
    dataset = BraTSSliceDataset(
        root_dir=data_cfg["path"],
        split=data_cfg["split"],
        img_size=img_size,
        split_txt_dir=data_cfg["split_txt"],
        skip_empty=False
    )

    # Group slice indices by subject
    print("🔎 Grouping slices by subject...")
    subject_to_slices = defaultdict(list)

    for i in range(len(dataset)):
        _, _, subject_id, slice_idx = dataset.samples[i]
        subject_to_slices[subject_id].append((i, slice_idx))

    print(f"📊 Total subjects to evaluate: {len(subject_to_slices)}")

    # Case-level evaluation
    dice_ET = dice_TC = dice_WT = 0
    N = 0

    for subject_id, slice_list in tqdm(subject_to_slices.items(), desc="Evaluating subjects", ncols=100):

        max_slice = max([z for (_, z) in slice_list])
        vol_pred = np.zeros((240, 240, max_slice + 1), dtype=np.uint8)
        vol_gt   = np.zeros_like(vol_pred)

        for dataset_idx, z in slice_list:
            img, lbl, _, _ = dataset[dataset_idx]

            img = img.unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(img)
            pred = torch.argmax(out, dim=1)[0].cpu().numpy()

            pred = F.interpolate(
                torch.from_numpy(pred)[None, None].float(),
                size=(240, 240),
                mode="nearest"
            )[0, 0].numpy()

            lbl = F.interpolate(
                torch.from_numpy(lbl.numpy())[None, None].float(),
                size=(240, 240),
                mode="nearest"
            )[0, 0].numpy()

            vol_pred[:, :, z] = pred
            vol_gt[:, :, z] = lbl

        # Dice
        et_pred = (vol_pred == 3)
        et_gt   = (vol_gt   == 3)

        tc_pred = np.isin(vol_pred, [2, 3])
        tc_gt   = np.isin(vol_gt, [2, 3])

        wt_pred = (vol_pred > 0)
        wt_gt   = (vol_gt > 0)

        dice_ET += dice_3d(et_pred, et_gt)
        dice_TC += dice_3d(tc_pred, tc_gt)
        dice_WT += dice_3d(wt_pred, wt_gt)
        N += 1

        del vol_pred, vol_gt
        torch.cuda.empty_cache()

    # -----------------------
    # Final output (case-level average)
    # -----------------------
    dET = dice_ET / N
    dTC = dice_TC / N
    dWT = dice_WT / N
    macro_avg = (dET + dTC + dWT) / 3

    print("\n================== RESULTS ==================")
    print(f"Dice_ET   : {dET:.4f}")
    print(f"Dice_TC   : {dTC:.4f}")
    print(f"Dice_WT   : {dWT:.4f}")
    print(f"Macro Avg : {macro_avg:.4f}")
    print("==============================================")

    return dET, dTC, dWT, macro_avg



# -----------------------
# Wrapper function (required by train_seg.py)
# -----------------------
def run_evaluation_2d_to_3d(config_path, checkpoint_path, output_dir):
    dET, dTC, dWT, macro_avg = evaluate_2d_to_3d(config_path, checkpoint_path)

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "3d_eval_results.txt")

    with open(out_file, "w") as f:
        f.write(f"Dice_ET: {dET:.4f}\n")
        f.write(f"Dice_TC: {dTC:.4f}\n")
        f.write(f"Dice_WT: {dWT:.4f}\n")
        f.write(f"Macro Avg: {macro_avg:.4f}\n")

    print(f"📄 3D evaluation results saved to {out_file}")
    return dET, dTC, dWT, macro_avg


# -----------------------
# Script Entry Point
# -----------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    print("🚀 Running 2D→3D evaluation...")
    evaluate_2d_to_3d(args.config, args.checkpoint)
    print("🏁 Evaluation complete!")
