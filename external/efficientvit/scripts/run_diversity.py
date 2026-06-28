#!/usr/bin/env python3
import os
import argparse
import numpy as np


# ============================================================
# Diversity Greedy (Sum of Pairwise Distances)
# ============================================================
def diversity_full_ranking(X, max_rank=750, normalize=True, seed=0, eps=1e-12):

    N, D = X.shape
    K = min(max_rank, N)

    X = X.astype(np.float64)

    if normalize:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

    print(f"🔹 Running Diversity Greedy (max_rank={K})...")

    rng = np.random.RandomState(seed)

    selected = []
    gains = []

    # 1️⃣ random start
    first = rng.randint(N)
    selected.append(first)

    # initial distance sum
    dist_sum = np.linalg.norm(X - X[first], axis=1)
    gains.append(float(dist_sum.max()))

    for step in range(1, K):

        idx = np.argmax(dist_sum)
        gain = dist_sum[idx]

        selected.append(idx)
        gains.append(float(gain))

        # update cumulative distance
        new_dist = np.linalg.norm(X - X[idx], axis=1)
        dist_sum += new_dist

        if step % 50 == 0:
            print(f"Step {step}/{K}")

    return selected, gains


# ============================================================
# Main
# ============================================================
def main():

    parser = argparse.ArgumentParser("Diversity for UPENN / IVYGAP / C5 / TCGA_LGG / TCGA_GBM")
    parser.add_argument("--target", required=True, choices=["UPENN", "IVYGAP", "C5", "TCGA_LGG", "TCGA_GBM"])
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--max_rank", type=int, default=750)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    embed_root = (
        "./"
        "./data/splits_"
        f"{args.target}_rds"
    )

    out_root = (
        "./"
        "./data/splits_"
        f"{args.target}_diversity"
    )

    os.makedirs(out_root, exist_ok=True)

    print(f"\n📂 Loading embeddings from: {embed_root}")

    src_vecs = np.load(os.path.join(embed_root, "src_subject_vecs.npy"))

    with open(os.path.join(embed_root, "src_subject_ids.txt")) as f:
        src_ids = [line.strip() for line in f]

    print(f"Source subjects: {len(src_ids)}")

    # ============================================================
    # 1️⃣ Diversity Ranking
    # ============================================================
    selected_order, gains = diversity_full_ranking(
        src_vecs,
        max_rank=args.max_rank,
        normalize=args.normalize,
        seed=args.seed
    )

    # ============================================================
    # 2️⃣ Build Marginal Gain-Based Score Dict
    # ============================================================
    score_dict = {sid: 0.0 for sid in src_ids}

    for idx, gain in zip(selected_order, gains):
        score_dict[src_ids[idx]] = float(gain)

    score_path = os.path.join(out_root, "diversity_score_dict.npy")
    np.save(score_path, score_dict, allow_pickle=True)

    print(f"💾 Saved diversity_score_dict.npy → {score_path}")

    # save greedy order
    ordered_ids = [src_ids[i] for i in selected_order]
    with open(os.path.join(out_root, "diversity_sorted_ids.txt"), "w") as f:
        f.write("\n".join(ordered_ids))

    print("💾 Saved greedy order.")

    # ============================================================
    # 3️⃣ Generate Budgets
    # ============================================================
    budgets_T = [1, 5, 10, 15]

    for k in budgets_T:

        budget = k * args.T
        subset_ids = ordered_ids[:budget]

        subset_dir = os.path.join(out_root, f"diversity_{k}T")
        os.makedirs(subset_dir, exist_ok=True)

        with open(os.path.join(subset_dir, "train_subjects.txt"), "w") as f:
            f.write("\n".join(subset_ids))

        print(f"✅ Saved diversity_{k}T ({budget})")

    print("\n🎉 Diversity completed.")


if __name__ == "__main__":
    main()
