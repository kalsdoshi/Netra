"""
Diagnose face similarity distribution and show top matching pairs.
Run: python scripts/diagnose.py
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from storage.store import FaceStore
from config import Config


def main():
    cfg   = Config()
    store = FaceStore(cfg)
    embs, meta = store.load_existing()

    if len(embs) == 0:
        print("No embeddings found. Run: python main.py --stage detect")
        return

    print(f"Total faces: {len(embs)}")

    # Full pairwise similarity matrix
    sim = embs @ embs.T
    np.fill_diagonal(sim, -1)  # exclude self-similarity

    best_sim = sim.max(axis=1)

    print("\n=== Nearest-neighbor similarity distribution ===")
    for t in [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]:
        count = int((best_sim >= t).sum())
        pct   = count / len(embs) * 100
        bar   = "█" * (count // 2)
        print(f"  sim >= {t:.2f}:  {count:3d} faces  ({pct:4.0f}%)  {bar}")

    print("\n=== Top 20 most similar cross-image pairs ===")
    rows, cols = np.triu_indices(len(embs), k=1)
    pair_sims  = sim[rows, cols]
    top_idx    = np.argsort(pair_sims)[::-1]

    def fname(path):
        return path.replace("\\", "/").split("/")[-1]

    printed = 0
    print(f"  {'Sim':>6}  {'Face A':<35}  {'Face B':<35}  Note")
    print("  " + "-" * 90)

    for idx in top_idx:
        if printed >= 20:
            break
        i, j  = int(rows[idx]), int(cols[idx])
        s      = float(pair_sims[idx])
        img_a  = fname(meta[i]["image_path"])
        img_b  = fname(meta[j]["image_path"])
        fi, fj = meta[i]["face_idx"], meta[j]["face_idx"]

        if img_a == img_b:
            note = "(same image)"
        elif s >= 0.70:
            note = "<-- likely same person"
        elif s >= 0.50:
            note = "<-- possible match"
        else:
            note = ""

        print(f"  {s:.4f}  {img_a+f'[{fi}]':<35}  {img_b+f'[{fj}]':<35}  {note}")
        printed += 1

    # Summary recommendation
    n_strong = int((best_sim >= 0.65).sum())
    print(f"\n=== Recommendation ===")
    print(f"  {n_strong} faces have a strong match (sim >= 0.65)")
    print(f"  Suggested epsilon: ", end="")

    if n_strong < 10:
        print("0.20  (few repeating faces — keep clusters tight)")
    elif n_strong < 30:
        print("0.35  (moderate repeats — default setting)")
    else:
        print("0.45  (many repeating faces — allow looser merging)")


if __name__ == "__main__":
    main()