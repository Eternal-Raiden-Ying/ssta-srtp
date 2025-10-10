#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_and_eval.py — 单图推理 + R² 评估（Windows / PyCharm 友好）

用法示例（外部图 .dgl）：
  python infer_and_eval.py ^
    --checkpoint_dir checkpoints\\08_atcd_specul ^
    --iter 15799 ^
    --graph D:\\work\\graphs\\b20_author_compat.dgl ^
    --nodes_csv D:\\work\\flow\\merge\\b20\\merged_nodes_inner_filtered.csv ^
    --out_dir D:\\work\\runs\\b20 ^
    --calibrate ^
    --plot

说明：
- 节点类型自动选择：优先 'node'，否则 'pin'。
- 边类型需要至少含 'cell_out'、'net_out'；缺 'net_in' 会自动补一份反向边（ef 同 net_out）。
- 会输出：
    1) 预测 CSV：<out_dir>\\pred_at.csv  （含 node_key, pred_at）
    2) 评估控制台日志（ALL / ENDPOINTS_EP 的 R² / r / MAE / RMSE）
    3) 可选散点图：<out_dir>\\scatter_all.png / scatter_ep.png
"""

import os, re, time, argparse, numpy as np, pandas as pd
from pathlib import Path

import torch, dgl

# ====== 你的模型 ======
from model import TimingGCN  # 保持与项目一致

# ---------------- 工具：图适配（基于你现有 infer_one 脚本） ----------------
def _device_of_graph(g: dgl.DGLHeteroGraph) -> torch.device:
    for nt in g.ntypes:
        for _, v in g.nodes[nt].data.items():
            if isinstance(v, torch.Tensor):
                return v.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _ensure_fields(g: dgl.DGLHeteroGraph, ntype: str):
    # n_net_delays_log 兜底
    if "n_net_delays_log" not in g.nodes[ntype].data and "n_net_delays" in g.nodes[ntype].data:
        g.nodes[ntype].data["n_net_delays_log"] = torch.log(1e-4 + g.nodes[ntype].data["n_net_delays"]) + 7.6
    # ef 统一 float32
    for et in g.etypes:
        if "ef" in g.edges[et].data:
            g.edges[et].data["ef"] = g.edges[et].data["ef"].to(torch.float32)
    # n_atslew 自动合成（若缺）
    if "n_atslew" not in g.nodes[ntype].data:
        if "n_ats" in g.nodes[ntype].data and "n_slews" in g.nodes[ntype].data:
            g.nodes[ntype].data["n_atslew"] = torch.cat(
                [g.nodes[ntype].data["n_ats"], torch.log(1e-4 + g.nodes[ntype].data["n_slews"]) + 3.0],
                dim=1
            )
        else:
            raise KeyError("需要 'n_ats'+'n_slews' 或已合成的 'n_atslew'。")

def ensure_net_in(g: dgl.DGLHeteroGraph, ntype: str) -> dgl.DGLHeteroGraph:
    """若缺 'net_in'，用 'net_out' 镜像补齐。"""
    if "net_out" in g.etypes and "net_in" not in g.etypes:
        u, v = g.edges(etype="net_out")
        data = {
            (ntype, "net_out", ntype): (u, v),
            (ntype, "net_in",  ntype): (v, u)
        }
        if "cell_out" in g.etypes:
            cu, cv = g.edges(etype="cell_out")
            data[(ntype, "cell_out", ntype)] = (cu, cv)
        ng = dgl.heterograph(data, num_nodes_dict={ntype: g.num_nodes(ntype)}).to(_device_of_graph(g))
        # 复制 ndata
        for k, v in g.nodes[ntype].data.items():
            ng.nodes[ntype].data[k] = v
        # 复制 edata
        ng.edges["net_out"].data["ef"] = g.edges["net_out"].data["ef"]
        ng.edges["net_in"].data["ef"]  = ng.edges["net_out"].data["ef"]
        if "cell_out" in g.etypes:
            ng.edges["cell_out"].data["ef"] = g.edges["cell_out"].data["ef"]
        return ng
    return g

def _reduce_corners(x: torch.Tensor, mode="max") -> torch.Tensor:
    if mode == "max":  return x.max(dim=1).values
    if mode == "min":  return x.min(dim=1).values
    return x.mean(dim=1)

def build_ts_fallback(g: dgl.DGLHeteroGraph, ntype: str):
    nf = g.nodes[ntype].data["nf"]; dev = nf.device
    is_pi = (nf[:,0] > 0.5); is_po = (nf[:,1] > 0.5)
    to_long = lambda m: m.nonzero(as_tuple=False).flatten().to(device=dev, dtype=torch.long)
    topo = []
    # 简单 topo：把 net_out + cell_out 当作有向边
    if "net_out" in g.etypes and "cell_out" in g.etypes:
        na, nb = g.edges(etype="net_out",  form="uv")
        ca, cb = g.edges(etype="cell_out", form="uv")
        hg = dgl.graph((torch.cat([na, ca]).cpu(), torch.cat([nb, cb]).cpu()))
        for t in dgl.topological_nodes_generator(hg):
            topo.append(t.to(dev))
    return {
        "input_nodes": to_long(~is_po),
        "output_nodes": to_long(is_po),
        "output_nodes_nonpi": to_long(is_po & (~is_pi)),
        "pi_nodes": to_long(is_pi),
        "po_nodes": to_long(is_po),
        "endpoints": to_long(g.nodes[ntype].data.get("n_is_timing_endpt", torch.zeros_like(is_pi)) > 0.5),
        "topo": topo,
        "topo_time": 0.0
    }

# ---------------- 评估（整合自你的 eval_r2 脚本） ----------------
PIN_D_SET = {
    "D","D_N","D0","D1","SD","SELD","DATA","DIN",
    "A","A0","A1","IN","I"
}
SEQ_CELL_PAT = re.compile(r"(dff|sdff|dffe|dfxtp|dfrtp|dfxbp|dfbbp|edfx|dlxtp|dlrtp|latch|lat)\b", re.I)
SEQ_TEXT_PAT = re.compile(r"(seq|clocked|next_state|ff|latch)", re.I)

def _norm_key(s):
    if pd.isna(s): return ""
    return str(s).strip().replace("\\[","[").replace("\\]","]").replace("\\/","/")

def load_nodes_sta(nodes_csv: str) -> pd.DataFrame:
    df = pd.read_csv(nodes_csv, low_memory=False)
    df.columns = [c.lower() for c in df.columns]

    if "node_key" not in df.columns:
        if {"inst","pin"}.issubset(df.columns):
            df["node_key"] = (df["inst"].map(_norm_key)+"/"+df["pin"].map(_norm_key))
        else:
            raise RuntimeError("nodes_csv 需要 node_key 或 inst+pin")

    df["node_key"] = df["node_key"].map(_norm_key)

    lr = pd.to_numeric(df.get("at_late_rise"), errors="coerce")
    lf = pd.to_numeric(df.get("at_late_fall"), errors="coerce")
    sta_at = np.nanmax(np.stack([lr, lf], axis=1), axis=1)

    is_primary_io = pd.to_numeric(df.get("is_primary_io", 0), errors="coerce").fillna(0).astype(int)
    dircol        = df.get("dir","").astype(str).str.lower()
    lib_pin       = df.get("lib_pin","").astype(str).str.upper()
    pin_timing    = df.get("pin_timing","").astype(str)
    inst_timing   = df.get("inst_timing","").astype(str)
    pin_only      = df.get("pin_only","").astype(str)
    ref_cell      = df.get("ref_cell","").astype(str)

    looks_seq_ref   = ref_cell.str.contains(SEQ_CELL_PAT)
    looks_seq_text1 = pin_timing.str.contains(SEQ_TEXT_PAT)
    looks_seq_text2 = inst_timing.str.contains(SEQ_TEXT_PAT)
    looks_seq_text3 = pin_only.str.contains(SEQ_TEXT_PAT)
    looks_seq_any   = looks_seq_ref | looks_seq_text1 | looks_seq_text2 | looks_seq_text3

    is_input  = dircol.eq("input")
    is_dname  = lib_pin.isin(PIN_D_SET) | df["node_key"].str.endswith("/D")
    is_ff_d   = is_input & is_dname & looks_seq_any
    is_po     = (is_primary_io==1) & dircol.eq("output")

    sta = pd.DataFrame({
        "node_key": df["node_key"],
        "sta_at":   sta_at,
        "is_endpoint_ep": (is_po | is_ff_d).astype(int),
        "is_primary_io":  is_primary_io,
        "dir": dircol
    })
    return sta

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y = y_true[m]; x = y_pred[m]
    N = len(y)
    if N == 0:
        return dict(N=0,R2=np.nan,r=np.nan,rho=np.nan,MAE=np.nan,RMSE=np.nan,alpha=None,beta=None)
    mse = np.mean((y-x)**2); var = np.mean((y-np.mean(y))**2)
    R2 = 1 - mse/var if var>0 else np.nan
    r  = np.corrcoef(y,x)[0,1] if N>1 else np.nan
    rho= np.corrcoef(pd.Series(y).rank(), pd.Series(x).rank())[0,1] if N>1 else np.nan
    return dict(N=N,R2=R2,r=r,rho=rho,MAE=np.mean(np.abs(y-x)),RMSE=np.sqrt(mse),alpha=None,beta=None)

def maybe_calibrate(y_true, y_pred):
    x = np.asarray(y_pred, float); y = np.asarray(y_true, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() <= 2: return dict(alpha=1.0, beta=0.0, y=x)
    A = np.vstack([x[m], np.ones(m.sum())]).T
    alpha, beta = np.linalg.lstsq(A, y[m], rcond=None)[0]
    y_cal = alpha * x + beta
    return dict(alpha=float(alpha), beta=float(beta), y=y_cal)

# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True, help="checkpoint 目录，比如 checkpoints\\08_xxx")
    ap.add_argument("--iter", type=int, required=True, help="迭代号，例如 15799")
    ap.add_argument("--graph", required=True, help=".dgl 图路径")
    ap.add_argument("--nodes_csv", required=True, help="merged_nodes_inner_filtered.csv 路径")
    ap.add_argument("--out_dir", required=True, help="输出目录")
    ap.add_argument("--corner", choices=["max","mean","min"], default="max")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 加载模型
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimingGCN().to(dev)
    ckpt_path = Path(args.checkpoint_dir) / f"{args.iter}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")
    try:
        state = torch.load(str(ckpt_path), map_location=dev, weights_only=True)
        model.load_state_dict(state)
    except TypeError:
        model.load_state_dict(torch.load(str(ckpt_path), map_location=dev))
    model.eval()

    # 2) 加载图（自适配 ntype & net_in）
    glist, _ = dgl.load_graphs(str(args.graph))
    if not glist:
        raise RuntimeError(f"无法读取图: {args.graph}")
    g = glist[0]
    ntype = "node" if "node" in g.ntypes else ("pin" if "pin" in g.ntypes else g.ntypes[0])
    g = g.to(dev)
    _ensure_fields(g, ntype)
    g = ensure_net_in(g, ntype)
    _ensure_fields(g, ntype)

    # 3) 构造 ts（外部图兜底）
    ts = build_ts_fallback(g, ntype)

    # 4) 推理
    if dev.type == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        pred_atslew = model(g, ts, groundtruth=False)[2][:, :4]  # [N, 4]
    if dev.type == "cuda": torch.cuda.synchronize()
    dt = time.time() - t0

    pred_at_1d = _reduce_corners(pred_atslew, args.corner)  # [N]
    print(f"\n[OK] Inference finished in {dt:.3f}s | graph={os.path.basename(args.graph)} | nodes={g.num_nodes(ntype)}")

    # 5) 生成 node_key（来自 nodes_csv，对齐 node_id）
    sta_df = load_nodes_sta(str(args.nodes_csv))
    node_keys = None
    if "node_id" in g.nodes[ntype].data:
        node_id = g.nodes[ntype].data["node_id"].detach().cpu().numpy().astype(int)
        # nodes_csv 必须至少与图同长度；假定 build_graph 时用同顺序
        if len(sta_df) >= len(node_id):
            node_keys = sta_df["node_key"].iloc[node_id].tolist()
        else:
            print("[WARN] nodes_csv 行数少于图节点数，pred.csv 将不写 node_key")

    # 6) 保存预测 CSV
    pred_csv = out_dir / "pred_at.csv"
    df_pred = pd.DataFrame({
        "node_id": np.arange(pred_at_1d.numel(), dtype=np.int32),
        "pred_at": pred_at_1d.detach().cpu().numpy().reshape(-1)
    })
    if node_keys is not None:
        df_pred["node_key"] = node_keys
    df_pred.to_csv(pred_csv, index=False)
    print(f"[OK] saved: {pred_csv}")

    # 7) 评估：合并 STA
    if node_keys is None:
        # 若没有 node_key，尝试通过 node_id 合并（需 nodes_csv 也有 node_id）
        print("[WARN] 无 node_key，尝试 node_id 合并（如果 nodes_csv 无 node_id 列，将只评估 ALL（可能为空））")
        sta_df = sta_df.reset_index(drop=True)
        sta_df["node_id"] = np.arange(len(sta_df), dtype=np.int32)
        merged = df_pred.merge(sta_df[["node_id","node_key","sta_at","is_endpoint_ep","is_primary_io","dir"]],
                               on="node_id", how="inner")
    else:
        merged = df_pred.merge(sta_df[["node_key","sta_at","is_endpoint_ep","is_primary_io","dir"]],
                               on="node_key", how="inner")

    print(f"[INFO] merged rows: {len(merged)}")

    def _report(df, title, do_calib=False, plot_name=None):
        res = metrics(df["sta_at"], df["pred_at"])
        print(f"\n=== {title} (raw) ===")
        for k in ["N", "R2", "r", "rho", "MAE", "RMSE"]:
            print(f"{k:>5}: {res[k]}")
        y_pred = df["pred_at"].to_numpy()
        if do_calib:
            cal = maybe_calibrate(df["sta_at"].to_numpy(), y_pred)
            y_use = cal["y"]
            res2 = metrics(df["sta_at"], y_use)
            print(f"[CALIB] y ≈ {cal['alpha']:.6f} * pred + {cal['beta']:.6f}")
            print(f"=== {title} (calibrated) ===")
            for k in ["N", "R2", "r", "rho", "MAE", "RMSE"]:
                print(f"{k:>5}: {res2[k]}")
        else:
            y_use = y_pred

        if plot_name and args.plot:
            import matplotlib.pyplot as plt
            X = df["sta_at"].to_numpy()
            plt.figure(figsize=(7, 5))
            plt.scatter(X, y_use, s=5, alpha=0.35)
            xx = np.linspace(np.nanmin(X), np.nanmax(X), 100)
            plt.plot(xx, xx, linewidth=2)
            head = metrics(X, y_use)
            plt.title(f"{title}  R²={head['R2']:.3f}, r={head['r']:.3f}, ρ={head['rho']:.3f}, N={head['N']}")
            plt.xlabel("STA AT (ns)");
            plt.ylabel("Pred AT (ns)")
            plt.tight_layout();
            plt.savefig(out_dir / plot_name, dpi=130)
            print(f"[OK] plot saved: {out_dir / plot_name}")

    # ALL
    _report(merged, "ALL", do_calib=args.calibrate, plot_name="scatter_all.png")
    # ENDPOINTS: PO ∪ FF_D
    ep = merged[merged["is_endpoint_ep"]==1]
    print(f"[INFO] endpoints_ep: {len(merged)} -> {len(ep)}")
    _report(ep, "ENDPOINTS_EP", do_calib=args.calibrate, plot_name="scatter_ep.png")

if __name__ == "__main__":
    main()
