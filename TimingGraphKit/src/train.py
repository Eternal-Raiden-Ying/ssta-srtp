#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finetune_single_graph.py — 在单张 .dgl 图上做快速微调（TimingGCN）

两种训练模式：
  1) --mode head  (默认)：只训练一个 4 通道的仿射标定头（y = a*x + b），极快、稳。
  2) --mode full  ：端到端训练 TimingGCN 全部参数（可配 --lr 小一些，或 --freeze_gnn）。

会保存：
  - best_finetune.pth         （仅头/或整模参数）
  - metrics.json              （train/val/test 的 R² / r / MAE / RMSE）
  - pred_{split}.csv          （各 split 的预测）

调用：
python finetune_single_graph.py ^
  --graph D:\...\b20_mod_fixed.graph.bin ^
  --nodes_csv D:\...\merge\b20\merged_nodes_inner_filtered.csv ^
  --ckpt_dir D:\...\checkpoints\08_atcd_specul ^
  --iter 15799 ^
  --out_dir D:\...\finetune\b20_head ^
  --mode head ^
  --epochs 150 ^
  --lr 5e-4 ^
  --endpoint_weight 2.0
"""

import os, json, argparse, random, numpy as np, pandas as pd
from pathlib import Path
import torch, torch.nn as nn
import dgl

# ==== 你的模型 ====
from model import TimingGCN

# ---------- 实用函数 ----------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_ntype(g):
    return "node" if "node" in g.ntypes else ("pin" if "pin" in g.ntypes else g.ntypes[0])

def ensure_fields(g, ntype):
    # ef->float32
    for et in g.etypes:
        if "ef" in g.edges[et].data:
            g.edges[et].data["ef"] = g.edges[et].data["ef"].to(torch.float32)
    # n_net_delays_log 兜底
    if "n_net_delays_log" not in g.nodes[ntype].data and "n_net_delays" in g.nodes[ntype].data:
        g.nodes[ntype].data["n_net_delays_log"] = torch.log(
            1e-4 + g.nodes[ntype].data["n_net_delays"].to(torch.float32)
        ) + 7.6
    # n_atslew 兜底
    if "n_atslew" not in g.nodes[ntype].data:
        if "n_ats" in g.nodes[ntype].data and "n_slews" in g.nodes[ntype].data:
            g.nodes[ntype].data["n_atslew"] = torch.cat(
                [g.nodes[ntype].data["n_ats"].to(torch.float32),
                 torch.log(1e-4 + g.nodes[ntype].data["n_slews"].to(torch.float32)) + 3.0],
                dim=1
            )
        else:
            raise KeyError("图缺少 'n_atslew'，且无法由 'n_ats'+'n_slews' 合成。")
    # 端点位兜底
    if "n_is_timing_endpt" not in g.nodes[ntype].data:
        g.nodes[ntype].data["n_is_timing_endpt"] = torch.zeros(
            (g.num_nodes(ntype), 1), dtype=torch.float32, device=g.device
        )

def ensure_net_in(g, ntype):
    if "net_out" in g.etypes and "net_in" not in g.etypes:
        u, v = g.edges(etype="net_out")
        data = {(ntype,"net_out",ntype):(u,v), (ntype,"net_in",ntype):(v,u)}
        if "cell_out" in g.etypes:
            cu, cv = g.edges(etype="cell_out")
            data[(ntype,"cell_out",ntype)] = (cu,cv)
        ng = dgl.heterograph(data, num_nodes_dict={ntype:g.num_nodes(ntype)}).to(g.device)
        # 复制 ndata/edata
        for k,v in g.nodes[ntype].data.items(): ng.nodes[ntype].data[k] = v
        ng.edges["net_out"].data["ef"] = g.edges["net_out"].data["ef"]
        ng.edges["net_in"].data["ef"]  = ng.edges["net_out"].data["ef"]
        if "cell_out" in g.etypes: ng.edges["cell_out"].data["ef"] = g.edges["cell_out"].data["ef"]
        return ng
    return g

def build_ts_from_graph(g, ntype):
    nf = g.nodes[ntype].data["nf"]; dev = nf.device
    is_pi = (nf[:,0] > 0.5); is_po = (nf[:,1] > 0.5)
    topo = []
    if "net_out" in g.etypes and "cell_out" in g.etypes:
        u1,v1 = g.edges(etype="net_out",  form="uv")
        u2,v2 = g.edges(etype="cell_out", form="uv")
        sg = dgl.graph((torch.cat([u1,u2]).cpu(), torch.cat([v1,v2]).cpu()))
        for t in dgl.topological_nodes_generator(sg):
            topo.append(t.to(dev))
    to_long = lambda m: m.nonzero(as_tuple=False).flatten().to(device=dev, dtype=torch.long)
    return {
        "input_nodes": to_long(~is_po),
        "output_nodes": to_long(is_po),
        "output_nodes_nonpi": to_long(is_po & (~is_pi)),
        "pi_nodes": to_long(is_pi),
        "po_nodes": to_long(is_po),
        "endpoints": to_long(g.nodes[ntype].data["n_is_timing_endpt"][:,0] > 0.5),
        "topo": topo,
        "topo_time": 0.0
    }

def reduce_corners(pred4, how="max"):
    if how == "max":  return pred4.max(1).values
    if how == "min":  return pred4.min(1).values
    return pred4.mean(1)

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred); y = y_true[m]; x = y_pred[m]
    N = y.size
    if N == 0: return dict(N=0,R2=np.nan,r=np.nan,MAE=np.nan,RMSE=np.nan)
    mse = np.mean((y-x)**2); var = np.mean((y-np.mean(y))**2)
    R2 = 1 - mse/var if var>0 else np.nan
    r  = np.corrcoef(y,x)[0,1] if N>1 else np.nan
    return dict(N=N,R2=R2,r=r,MAE=np.mean(np.abs(y-x)),RMSE=np.sqrt(mse))

def load_sta(nodes_csv):
    df = pd.read_csv(nodes_csv, low_memory=False)
    if "node_key" not in df.columns and {"inst","pin"}.issubset(df.columns):
        df["node_key"] = df["inst"].astype(str)+"/"+df["pin"].astype(str)
    lr = pd.to_numeric(df.get("at_late_rise"), errors="coerce")
    lf = pd.to_numeric(df.get("at_late_fall"), errors="coerce")
    sta_at = np.nanmax(np.stack([lr, lf], axis=1), axis=1)
    return df.assign(sta_at=sta_at)[["node_key","sta_at"]]

# ---------- 标定头（4通道仿射） ----------
class AffineHead(nn.Module):
    def __init__(self, channels=4):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(channels))
        self.beta  = nn.Parameter(torch.zeros(channels))
    def forward(self, x):  # x: [N,4]
        return x * self.alpha + self.beta

# ---------- 主训练 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--nodes_csv", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", choices=["head","full"], default="head")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--corner", choices=["max","mean","min"], default="max")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--endpoint_weight", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 设备
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型
    model = TimingGCN().to(dev)
    ckpt = Path(args.ckpt_dir) / f"{args.iter}.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt}")
    try:
        state = torch.load(str(ckpt), map_location=dev, weights_only=True)
        model.load_state_dict(state)
    except TypeError:
        model.load_state_dict(torch.load(str(ckpt), map_location=dev))
    model.train()

    # 图
    glist,_ = dgl.load_graphs(args.graph)
    g = glist[0].to(dev)
    ntype = get_ntype(g)
    g = ensure_net_in(g, ntype); ensure_fields(g, ntype)
    ts = build_ts_from_graph(g, ntype)

    # 头（仅 head 模式训练；full 模式则把头设为恒等）
    head = AffineHead(4).to(dev)
    if args.mode == "full":
        # 给个轻微初值（可不设）：保持一开始是恒等
        head.alpha.data[:] = 1.0; head.beta.data[:] = 0.0

    # 标签（与 node_id 对齐）
    sta = load_sta(args.nodes_csv)
    node_id = g.nodes[ntype].data.get("node_id", torch.arange(g.num_nodes(ntype))).cpu().numpy()
    if len(sta) >= len(node_id):
        sta = sta.iloc[node_id].reset_index(drop=True)
    else:
        sta = sta.iloc[:len(node_id)].reset_index(drop=True)
    y = torch.tensor(sta["sta_at"].to_numpy(), dtype=torch.float32, device=dev)  # [N]

    # 可训练参数
    if args.mode == "head":
        for p in model.parameters(): p.requires_grad = False
        optim = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        params = list(model.parameters()) + list(head.parameters())
        optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # 划分 train/val/test（只在有标签的节点里）
    has = torch.isfinite(y)
    idx = torch.nonzero(has, as_tuple=False).flatten()
    Nlab = idx.numel()
    perm = idx[torch.randperm(Nlab)]
    n_test = int(Nlab * args.test_ratio)
    n_val  = int(Nlab * args.val_ratio)
    n_train = Nlab - n_val - n_test
    idx_train = perm[:n_train]; idx_val = perm[n_train:n_train+n_val]; idx_test = perm[n_train+n_val:]
    print(f"[Split] train={len(idx_train)}  val={len(idx_val)}  test={len(idx_test)}")

    # 端点加权
    ep_mask = (g.nodes[ntype].data["n_is_timing_endpt"][:,0] > 0.5)
    w = torch.ones(g.num_nodes(ntype), device=dev)
    w[ep_mask] = args.endpoint_weight

    # 训练循环（全图前向；单图通常能放得下）
    best_val = -1e9; patience, noimp = 20, 0
    history = []
    for epoch in range(1, args.epochs+1):
        model.train()
        optim.zero_grad()
        with torch.no_grad() if args.mode=="head" else torch.enable_grad():
            # 注意：head 总是要训练的；full 模式里 model 也训练
            pass

        pred4 = model(g, ts, groundtruth=False)[2][:,:4]   # [N,4]
        pred4 = head(pred4)                                # 标定头
        pred1 = reduce_corners(pred4, args.corner)         # [N]
        # 训练损失（仅训练集）
        err = (pred1[idx_train] - y[idx_train])
        loss = (w[idx_train] * err**2).mean()

        loss.backward(); optim.step()

        # 验证
        model.eval()
        with torch.no_grad():
            pred4 = model(g, ts, groundtruth=False)[2][:,:4]
            pred4 = head(pred4)
            pred1 = reduce_corners(pred4, args.corner)
            train_m = metrics(y[idx_train].detach().cpu().numpy(), pred1[idx_train].detach().cpu().numpy())
            val_m   = metrics(y[idx_val].detach().cpu().numpy(),   pred1[idx_val].detach().cpu().numpy())
            test_m  = metrics(y[idx_test].detach().cpu().numpy(),  pred1[idx_test].detach().cpu().numpy())
        history.append(dict(epoch=epoch, train=train_m, val=val_m, test=test_m))
        print(f"[{epoch:03d}] loss={float(loss):.4f}  "
              f"R2(train/val/test)={train_m['R2']:.3f}/{val_m['R2']:.3f}/{test_m['R2']:.3f}  "
              f"r(val)={val_m['r']:.3f}")

        # 早停
        if val_m["R2"] > best_val + 1e-5:
            best_val = val_m["R2"]; noimp = 0
            # 保存最好
            save = dict(mode=args.mode, head=dict(alpha=head.alpha.detach().cpu().numpy().tolist(),
                                                  beta=head.beta.detach().cpu().numpy().tolist()))
            if args.mode == "full":
                save["model"] = model.state_dict()
            torch.save(save, out_dir / "best_finetune.pth")
        else:
            noimp += 1
            if noimp >= patience:
                print(f"[EarlyStop] patience={patience}, best_val_R2={best_val:.3f}")
                break

    # 载入最佳并生成最终输出
    ck = torch.load(out_dir / "best_finetune.pth", map_location=dev)
    with torch.no_grad():
        head.alpha[:] = torch.tensor(ck["head"]["alpha"], device=dev)
        head.beta[:]  = torch.tensor(ck["head"]["beta"], device=dev)
        if args.mode == "full" and "model" in ck:
            model.load_state_dict(ck["model"])

        pred4 = model(g, ts, groundtruth=False)[2][:,:4]
        pred4 = head(pred4)
        pred1 = reduce_corners(pred4, args.corner).detach().cpu().numpy()

    # 输出 CSV & 指标
    node_id = g.nodes[ntype].data.get("node_id", torch.arange(g.num_nodes(ntype))).cpu().numpy()
    sta_np = sta = load_sta(args.nodes_csv)
    if len(sta) >= len(node_id): sta_np = sta.iloc[node_id].reset_index(drop=True)
    else:                        sta_np = sta.iloc[:len(node_id)].reset_index(drop=True)
    sta_np["pred_at"] = pred1
    sta_np["split"] = "train"
    sta_np.loc[idx_val.cpu().numpy(),  "split"] = "val"
    sta_np.loc[idx_test.cpu().numpy(), "split"] = "test"

    for split in ["train","val","test"]:
        df = sta_np[sta_np["split"]==split][["node_key","sta_at","pred_at"]]
        df.to_csv(out_dir / f"pred_{split}.csv", index=False)

    all_metrics = {
        "best_val_R2": best_val,
        "last": history[-1] if history else {},
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] saved to: {out_dir}")
    print(f" best_val_R2 = {best_val:.4f}")
    print(f" files: best_finetune.pth, pred_train/val/test.csv, metrics.json")

if __name__ == "__main__":
    main()
