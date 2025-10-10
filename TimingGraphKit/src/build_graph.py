#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, re
from pathlib import Path
import numpy as np
import pandas as pd
import dgl
import torch

def norm(s: str) -> str:
    if pd.isna(s): return ""
    s = str(s).strip()
    return (s.replace("\\[", "[").replace("\\]", "]").replace("\\/", "/")
              .replace("\\$", "$"))

def build_node_key(inst: str, pin: str) -> str:
    inst, pin = norm(inst), norm(pin)
    if inst and inst.upper() != "TOP":
        return f"{inst}/{pin}"
    return pin  # 顶层 IO

def finite_or_nan(arr, big=1e38):
    x = np.asarray(arr, dtype=np.float64)
    x[(np.abs(x)>=big) | (~np.isfinite(x))] = np.nan
    return x

def to_f32(x): return torch.as_tensor(x, dtype=torch.float32)

# -------- 顺序弧过滤启发式 --------
_SEQ_CELL_PAT = re.compile(r"(DFF|SDFF|DFFE|LATCH|LAT|TLAT|FF)\b", re.I)
_Q_PINS = {"Q","QN","QB","Q_N","QBAR","Q1","Q2","Q0","QN1","QN2"}
_CTRL_PAT = re.compile(r"(CLK|CK|CP|GATE|RESET|RST|SET|SN|RN|SE|TE|CE|CEN|SR|ACLR|ASET)", re.I)

def is_sequential_arc(row):
    # 需要列：from_pin/to_pin；如有 cell 列更准确
    to_pin = str(row.get("to_pin",""))
    from_pin = str(row.get("from_pin",""))
    cell = str(row.get("cell",""))
    # 输出脚像 Q/QN
    if to_pin in _Q_PINS: return True
    # 控制脚作为输入
    if _CTRL_PAT.search(from_pin): return True
    # 单元名里带 DFF/LATCH 等
    if cell and _SEQ_CELL_PAT.search(cell): return True
    return False

def parse_list_cell(v, target_len=None):
    if pd.isna(v): arr=[]
    else:
        s=str(v).strip()
        if s.startswith('[') and s.endswith(']'):
            try:
                import ast; arr=list(ast.literal_eval(s))
            except Exception:
                arr=re.split(r'[,\s]+', s.strip('[]'))
        else:
            arr=re.split(r'[,\s]+', s)
    arr=[float(x) for x in arr if str(x)!='']
    if (target_len is not None) and (len(arr)!=target_len):
        arr = (arr + [0.0]*target_len)[:target_len]
    return arr

def build_graph(
    root, platform, design,
    net_csv=None, cell_feat_csv=None, cell_tasks_csv=None,
    dist_csv=None,
    out_path="out.dgl",
    include_reverse=False,
    drop_seq_arcs=True,
    print_mapping=False
):
    # ------- 路径 -------
    NODES_CSV      = f"{root}/merge/{design}/merged_nodes_inner_filtered.csv"
    NET_EDGES_CSV  = net_csv       or f"{root}/edge_feats/{platform}/{design}/net_edges.csv"
    CELL_FEAT_CSV  = cell_feat_csv or f"{root}/edge_feats/{platform}/{design}/cell_edges.csv"
    CELL_TASKS_CSV = cell_tasks_csv or f"{root}/timing_task/{design}/edge_features/{design}.cell_arcs_delay.csv"
    DIST_CSV       = dist_csv

    print(f"[Path] nodes    : {NODES_CSV}")
    print(f"[Path] net_edges: {NET_EDGES_CSV}")
    print(f"[Path] cell_feat: {CELL_FEAT_CSV}")
    print(f"[Path] cell_task: {CELL_TASKS_CSV}")
    print(f"[Path] dist_csv : {DIST_CSV or '(none)'}")

    # ------- nodes -------
    nodes = pd.read_csv(NODES_CSV, low_memory=False)
    nodes.columns = [c.strip() for c in nodes.columns]
    if "node_key" in nodes.columns:
        nodes["node_key"] = nodes["node_key"].astype(str).map(norm)
    else:
        inst_col = next((c for c in nodes.columns if c.lower().startswith("inst")), None)
        pin_col  = next((c for c in nodes.columns if c.lower().startswith("pin")),  None)
        if not (inst_col and pin_col):
            raise RuntimeError("nodes CSV 缺少 node_key 或 inst/pin 列")
        nodes["node_key"] = [build_node_key(nodes[inst_col][i], nodes[pin_col][i]) for i in range(len(nodes))]

    node_keys = nodes["node_key"].tolist()
    nid = {k:i for i,k in enumerate(node_keys)}
    N = len(node_keys)

    dir_col = "dir" if "dir" in nodes.columns else None
    is_port_col = "is_primary_io" if "is_primary_io" in nodes.columns else None

    cap_cols = [c for c in nodes.columns if c.lower() in ("cap_er_ff","cap_ef_ff","cap_lr_ff","cap_lf_ff")]
    if len(cap_cols)!=4:
        alts = []
        for base in ["cap_er","cap_ef","cap_lr","cap_lf"]:
            cand = [c for c in nodes.columns if c.lower().startswith(base)]
            if cand: alts.append(cand[0])
        cap_cols = alts if len(alts)==4 else None

    at_cols   = [c for c in nodes.columns if c.lower().startswith("at_")]
    slew_cols = [c for c in nodes.columns if c.lower().startswith("slew_")]
    rat_cols  = [c for c in nodes.columns if c.lower().startswith("rat_")]

    # ------- merge dist4 -------
    if DIST_CSV and os.path.exists(DIST_CSV):
        pos = pd.read_csv(DIST_CSV, low_memory=False)
        pos.columns = [c.strip().lower() for c in pos.columns]
        need = {"inst","pin","dist_left","dist_right","dist_bottom","dist_top"}
        if need.issubset(pos.columns):
            pos["node_key"] = [build_node_key(pos["inst"][i], pos["pin"][i]) for i in range(len(pos))]
            nodes = nodes.merge(
                pos[["node_key","dist_left","dist_right","dist_bottom","dist_top"]],
                on="node_key", how="left")
        else:
            print("[WARN] dist_csv 列不全，已忽略。")

    # ------- net_edges -------
    net = pd.read_csv(NET_EDGES_CSV, low_memory=False)
    net.columns = [c.strip() for c in net.columns]
    if {"src","dst"}.issubset(net.columns):
        net["src"] = net["src"].astype(str).map(norm)
        net["dst"] = net["dst"].astype(str).map(norm)
    elif {"driver_inst","driver_pin","sink_inst","sink_pin"}.issubset(net.columns):
        net["src"] = [build_node_key(i,p) for i,p in zip(net["driver_inst"], net["driver_pin"])]
        net["dst"] = [build_node_key(i,p) for i,p in zip(net["sink_inst"],   net["sink_pin"])]
    else:
        raise RuntimeError("net_edges.csv 需 (src,dst) 或 (driver_inst,driver_pin,sink_inst,sink_pin)")

    def pick(cands):
        for c in cands:
            if c in net.columns: return c
        return None
    dx_um_col = pick(["dx_um","net_dx_um","dx"])
    dy_um_col = pick(["dy_um","net_dy_um","dy"])
    net["ef_dx"] = pd.to_numeric(net[dx_um_col], errors="coerce").fillna(0.0) if dx_um_col else 0.0
    net["ef_dy"] = pd.to_numeric(net[dy_um_col], errors="coerce").fillna(0.0) if dy_um_col else 0.0

    dst_in_nodes = net["dst"].isin(nid)
    if print_mapping:
        miss_dst = (~dst_in_nodes).sum()
        miss_src = (~net["src"].isin(nid)).sum()
        print(f"[Map] net_edges total={len(net)}  keep_by_dst_only={dst_in_nodes.sum()}  missing_src={miss_src}  missing_dst={miss_dst}")
        if miss_dst:
            print("  [sample missing dst]:", net.loc[~dst_in_nodes,"dst"].head(10).tolist())

    net_keep = net[dst_in_nodes].copy()
    u_series = net_keep["src"].map(nid)    # 可能有 NaN
    v_series = net_keep["dst"].map(nid)
    both_ok = u_series.notna() & v_series.notna()
    net_keep_ok = net_keep.loc[both_ok].copy()

    # 丢弃自环
    u_tmp = u_series[both_ok].astype(int).to_numpy()
    v_tmp = v_series[both_ok].astype(int).to_numpy()
    non_self = (u_tmp != v_tmp)
    if print_mapping and (~non_self).sum():
        print(f"[Filter] drop net self-loops: {(~non_self).sum()}")
    u_net = u_tmp[non_self]
    v_net = v_tmp[non_self]
    net_ef = np.stack([
        pd.to_numeric(net_keep_ok["ef_dx"].to_numpy(), errors="coerce").astype(np.float64)[non_self],
        pd.to_numeric(net_keep_ok["ef_dy"].to_numpy(), errors="coerce").astype(np.float64)[non_self]
    ], axis=1)

    # ------- cell_edges（只保留组合弧 + 去自环）-------
    cf = pd.read_csv(CELL_FEAT_CSV, low_memory=False)
    cf.columns = [c.strip().lower() for c in cf.columns]
    need_cf = {"inst","from_pin","to_pin"}
    if not need_cf.issubset(cf.columns):
        alias = {
            "inst":     next((c for c in cf.columns if c in ("cell_inst","instance","inst_name")), None),
            "from_pin": next((c for c in cf.columns if c in ("from","src_pin","in_pin")), None),
            "to_pin":   next((c for c in cf.columns if c in ("to","dst_pin","out_pin")), None),
        }
        if not all(alias.values()):
            raise RuntimeError("cell_edges.csv 缺 inst/from_pin/to_pin（或同义列）")
        cf["inst"], cf["from_pin"], cf["to_pin"] = cf[alias["inst"]], cf[alias["from_pin"]], cf[alias["to_pin"]]

    cf["src"] = [build_node_key(i,p) for i,p in zip(cf["inst"], cf["from_pin"])]
    cf["dst"] = [build_node_key(i,p) for i,p in zip(cf["inst"], cf["to_pin"])]
    src_ok = cf["src"].isin(nid); dst_ok = cf["dst"].isin(nid)
    if print_mapping:
        print(f"[Map] cell_edges total={len(cf)}  src_in_nodes={src_ok.sum()} dst_in_nodes={dst_ok.sum()}")

    cf_keep = cf[src_ok & dst_ok].copy()

    # 过滤顺序弧（启发式）
    dropped_seq = 0
    if drop_seq_arcs:
        mask_seq = cf_keep.apply(is_sequential_arc, axis=1)
        dropped_seq = int(mask_seq.sum())
        cf_keep = cf_keep[~mask_seq].copy()
        if print_mapping:
            print(f"[Filter] drop sequential arcs: {dropped_seq}")

    # 去自环（from_pin==to_pin / 节点重合）
    u_cell_series = cf_keep["src"].map(nid)
    v_cell_series = cf_keep["dst"].map(nid)
    u_cell_tmp = u_cell_series.astype(int).to_numpy()
    v_cell_tmp = v_cell_series.astype(int).to_numpy()
    non_self_cell = (u_cell_tmp != v_cell_tmp)
    if print_mapping and (~non_self_cell).sum():
        print(f"[Filter] drop cell self-loops: {(~non_self_cell).sum()}")
    u_cell = u_cell_tmp[non_self_cell]
    v_cell = v_cell_tmp[non_self_cell]

    # 打包 cell ef: 512 维，120:512 为 392 个 LUT 值
    lut_blocks = [
        "late_delay_rise","late_delay_fall","late_slew_rise","late_slew_fall",
        "early_delay_rise","early_delay_fall","early_slew_rise","early_slew_fall"
    ]
    cf_keep2 = cf_keep.iloc[non_self_cell].copy()
    vals_all = []
    for tag in lut_blocks:
        col = f"{tag}_values"
        if col in cf_keep2.columns:
            vals = np.vstack([parse_list_cell(v, 49) for v in cf_keep2[col]])
        else:
            vals = np.zeros((len(cf_keep2),49), dtype=np.float32)
        vals_all.append(vals)
    lut_392 = np.concatenate(vals_all, axis=1).astype(np.float32) if len(vals_all)>0 else np.zeros((len(cf_keep2),392), dtype=np.float32)
    ef_cell = np.zeros((len(cf_keep2), 512), dtype=np.float32)
    ef_cell[:,120:512] = lut_392

    # ------- 构图（是否带反向边可选）-------
    data_dict = {
        ('pin','net_out','pin')  : (u_net,  v_net),
        ('pin','cell_out','pin') : (u_cell, v_cell),
    }
    if include_reverse:
        data_dict[('pin','net_in','pin')]   = (v_net,  u_net)
        data_dict[('pin','cell_in','pin')]  = (v_cell, u_cell)

    g = dgl.heterograph(data_dict, num_nodes_dict={'pin': N})

    # ------- 节点特征（TimingPredict 命名习惯）-------
    if is_port_col and is_port_col in nodes:
        is_pi = nodes[is_port_col].astype(float).to_numpy()
    else:
        is_pi = nodes["node_key"].map(lambda k: 0.0 if "/" in k else 1.0).to_numpy(dtype=float)

    if dir_col and ("output" in set(nodes[dir_col].astype(str).str.lower())):
        is_po = (nodes[dir_col].astype(str).str.lower()=="output").astype(float).to_numpy()
    else:
        is_po = np.zeros(N, dtype=float)

    have_dist = all(c in nodes.columns for c in ["dist_left","dist_right","dist_bottom","dist_top"])
    dist4 = nodes[["dist_left","dist_right","dist_bottom","dist_top"]].to_numpy(dtype=np.float32) if have_dist \
        else np.zeros((N,4), dtype=np.float32)

    if cap_cols is not None and len(cap_cols)==4:
        cap4 = nodes[cap_cols].to_numpy(dtype=np.float32)
    else:
        cap4 = np.zeros((N,4), dtype=np.float32)

    nf = np.concatenate([
        is_pi.reshape(-1,1).astype(np.float32),
        is_po.reshape(-1,1).astype(np.float32),
        dist4, cap4
    ], axis=1)

    n_ats   = nodes[[c for c in at_cols]].to_numpy(dtype=np.float32)   if len(at_cols)>=4   else np.zeros((N,4), np.float32)
    n_slews = nodes[[c for c in slew_cols]].to_numpy(dtype=np.float32) if len(slew_cols)>=4 else np.zeros((N,4), np.float32)
    n_atslew = np.concatenate([n_ats, np.log(n_slews+1e-4)+3.0], axis=1).astype(np.float32)

    if all((f"net_delay_{k}" in nodes.columns) for k in ["er","ef","lr","lf"]):
        nd4 = nodes[[f"net_delay_{k}" for k in ["er","ef","lr","lf"]]].to_numpy(dtype=np.float32)
    else:
        nd4 = np.zeros((N,4), dtype=np.float32)
    nd4_log = (np.log(nd4+1e-4)+7.6).astype(np.float32)

    if len(rat_cols)>=4:
        rat4 = nodes[[c for c in rat_cols]].to_numpy(dtype=np.float32)
        is_endpt = (np.isfinite(finite_or_nan(rat4)).any(axis=1)).astype(np.float32).reshape(-1,1)
    else:
        rat4 = np.zeros((N,4), np.float32)
        is_endpt = np.zeros((N,1), np.float32)

    g.nodes['pin'].data['nf']                   = to_f32(nf)                 # 10
    g.nodes['pin'].data['n_ats']                = to_f32(n_ats)              # 4
    g.nodes['pin'].data['n_slews']              = to_f32(n_slews)            # 4
    g.nodes['pin'].data['n_atslew']             = to_f32(n_atslew)           # 8
    g.nodes['pin'].data['n_net_delays']         = to_f32(nd4)                # 4
    g.nodes['pin'].data['n_net_delays_log']     = to_f32(nd4_log)            # 4
    g.nodes['pin'].data['n_is_timing_endpt']    = to_f32(is_endpt)           # 1
    g.nodes['pin'].data['node_id']              = torch.arange(N, dtype=torch.int32)

    # 边特征
    g.edges[('pin','net_out','pin')].data['ef']  = to_f32(net_ef)   # 2D
    g.edges[('pin','cell_out','pin')].data['ef'] = to_f32(ef_cell)  # 512D
    if include_reverse:
        g.edges[('pin','net_in','pin')].data['ef']   = to_f32(net_ef)
        g.edges[('pin','cell_in','pin')].data['ef']  = to_f32(ef_cell)

    # ------- 自检 -------
    # ① cell 输入是否都被 net_out 驱动
    cell_inputs_all = cf["src"].unique().tolist()
    cell_inputs_all = [k for k in cell_inputs_all if k in nid]
    pid_inputs = torch.tensor([nid[k] for k in cell_inputs_all], dtype=torch.int64)
    indeg = g.in_degrees(pid_inputs, etype=('pin','net_out','pin')).numpy()
    if print_mapping:
        miss_count = int((indeg==0).sum())
        print(f"[Check] cell 输入 pin 总数={len(pid_inputs)}；其中 net_out 入度==0 的数量={miss_count}")

    # ② 统计边数
    e_counts = {('pin','net_out','pin'): g.num_edges(('pin','net_out','pin')),
                ('pin','cell_out','pin'): g.num_edges(('pin','cell_out','pin'))}
    if include_reverse:
        e_counts[('pin','net_in','pin')]  = g.num_edges(('pin','net_in','pin'))
        e_counts[('pin','cell_in','pin')] = g.num_edges(('pin','cell_in','pin'))

    # ------- 保存 -------
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    dgl.save_graphs(out_path, [g])
    print(f"[OK] saved: {out_path}")
    if include_reverse:
        print(f"[Edges] net_out: {e_counts[('pin','net_out','pin')]} net_in: {e_counts[('pin','net_in','pin')]} "
              f"cell_out: {e_counts[('pin','cell_out','pin')]} cell_in: {e_counts[('pin','cell_in','pin')]}")
    else:
        print(f"[Edges] net_out: {e_counts[('pin','net_out','pin')]} cell_out: {e_counts[('pin','cell_out','pin')]}")

    print(g)
    return g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--platform", required=True)
    ap.add_argument("--design", required=True)
    ap.add_argument("--net-csv", default=None)
    ap.add_argument("--cell-feat-csv", default=None)
    ap.add_argument("--cell-tasks-csv", default=None)
    ap.add_argument("--dist-csv", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-reverse", action="store_true", help="输出 net_in/cell_in 反向边（默认关闭）")
    ap.add_argument("--no-drop-seq-arcs", action="store_true", help="不要过滤顺序弧（默认会过滤）")
    ap.add_argument("--print-mapping", action="store_true")
    args = ap.parse_args()

    _ = build_graph(
        root=args.root,
        platform=args.platform,
        design=args.design,
        net_csv=args.net_csv,
        cell_feat_csv=args.cell_feat_csv,
        cell_tasks_csv=args.cell_tasks_csv,
        dist_csv=args.dist_csv,
        out_path=args.out,
        include_reverse=args.include_reverse,
        drop_seq_arcs=not args.no_drop_seq_arcs,
        print_mapping=args.print_mapping
    )

if __name__ == "__main__":
    main()
