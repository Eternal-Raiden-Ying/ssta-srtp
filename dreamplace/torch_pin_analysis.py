
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
torch_pin_analysis.py
Usage:
  python torch_pin_analysis.py /path/to/pins.csv [--topk 50]

The CSV is expected to have columns like:
inst,pin,x,y,dist_left,dist_right,dist_bottom,dist_top

What it does:
- Loads the CSV (pandas)
- Converts numeric columns to PyTorch tensors
- Prints summary stats (mean/std/min/max) for each numeric column
- Computes a correlation matrix
- Finds the TOP-K pins closest to each die edge
- Plots histograms for x, y, and each distance column (matplotlib)

Outputs:
- <csv_dir>/edge_topk_left.csv (and *_right/_bottom/_top)
- <csv_dir>/hist_x.png, hist_y.png, hist_dist_left.png, ...
"""
import argparse
import os
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', type=str, help='Path to pins.csv')
    parser.add_argument('--topk', type=int, default=50, help='Top-K nearest pins to each edge')
    args = parser.parse_args()

    csv_path = args.csv_path
    out_dir = os.path.dirname(os.path.abspath(csv_path)) or "."

    # 1) Load CSV
    df = pd.read_csv(csv_path)

    # 2) Select numeric columns & convert to torch
    numeric_cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
    if not numeric_cols:
        raise RuntimeError("No numeric columns found in the CSV.")
    t = torch.tensor(df[numeric_cols].to_numpy(), dtype=torch.float32)  # [N, D]
    col_index = {c:i for i,c in enumerate(numeric_cols)}

    # 3) Summary stats
    means = t.mean(dim=0)
    stds  = t.std(dim=0, unbiased=False)
    mins  = t.min(dim=0).values
    maxs  = t.max(dim=0).values

    print("=== Summary (DBU units) ===")
    for i, c in enumerate(numeric_cols):
        print(f"{c:>12s} | mean={means[i]:.3f} std={stds[i]:.3f} min={mins[i]:.3f} max={maxs[i]:.3f}")

    # 4) Correlation matrix (Pearson)
    corr = pd.DataFrame(df[numeric_cols]).corr(method='pearson')
    corr_path = os.path.join(out_dir, "corr_numeric.csv")
    corr.to_csv(corr_path, index=True)
    print(f"Saved correlation matrix -> {corr_path}")

    # 5) Top-K nearest to each edge
    def topk_by(colname, k):
        if colname not in df.columns:
            return None, None
        s = torch.tensor(df[colname].to_numpy(), dtype=torch.float32)
        k = min(k, s.numel())
        vals, idx = torch.topk(-s, k)  # negative for ascending (nearest)
        idx = idx.tolist()
        out = df.loc[idx, :].copy()
        out["rank_by_"+colname] = list(range(1, k+1))
        return out, colname

    edge_cols = ["dist_left", "dist_right", "dist_bottom", "dist_top"]
    for ec in edge_cols:
        if ec in df.columns:
            out_df, name = topk_by(ec, args.topk)
            if out_df is not None:
                out_path = os.path.join(out_dir, f"edge_topk_{name}.csv")
                out_df.to_csv(out_path, index=False)
                print(f"Saved Top-{args.topk} nearest to {name} -> {out_path}")

    # 6) Histograms for x, y, and distances
    def plot_hist(colname, bins=80):
        if colname not in df.columns:
            return
        plt.figure(figsize=(7,4.5))
        plt.hist(df[colname].to_numpy(), bins=bins)
        plt.xlabel(colname + " (DBU)")
        plt.ylabel("count")
        plt.title(f"Histogram of {colname}")
        out_path = os.path.join(out_dir, f"hist_{colname}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")

    for c in ["x", "y", "dist_left", "dist_right", "dist_bottom", "dist_top"]:
        plot_hist(c)

    print("Done.")

if __name__ == "__main__":
    main()
