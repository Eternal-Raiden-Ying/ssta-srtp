import os
import time
import logging
import argparse
from pathlib import Path

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score

from model import TimingGCN
from Mydataset import DGLGraphDataset, get_dataloder
from utils import *


# -----------------------------
# Utils
# -----------------------------
def collect_graph_bins(root_dir: str):
    """
    递归遍历 root_dir 下所有 *.graph.bin 文件：
    - dgl_graphs_path: 这些文件的绝对路径（list[str]）
    - labels: 去掉 .graph.bin 后的文件名（list[str]）
    """
    root = Path(root_dir).expanduser().resolve()
    dgl_graphs_path, labels = [], []
    for p in sorted(root.rglob("*.graph.bin")):
        dgl_graphs_path.append(str(p.resolve()))
        labels.append(p.name[:-len(".graph.bin")])
    return dgl_graphs_path, labels


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# -----------------------------
# Parser（精简后）
# -----------------------------
parser = argparse.ArgumentParser(description="TimingPredict-GNN (single Adam + LR policy)")

# 运行与设备
parser.add_argument("--device", type=str, default="cuda:0", help="device")
parser.add_argument("--epochs", type=int, default=1000, help="training epochs")
parser.add_argument("--start-epoch", type=int, default=0, help="start epoch")
parser.add_argument("--batch-size", type=int, default=1, help="batch size (建议保持 1)")
parser.add_argument("--num-workers", type=int, default=0, help="dataloader workers")
parser.add_argument("--pin-mem", type=bool, default=False, help="pin memory")

# 数据与输出
parser.add_argument("--data-root", type=str, required=True, help="包含 *.graph.bin 的根目录")
parser.add_argument("--output-dir", type=str, default="res", help="输出根目录")
parser.add_argument("--checkpoint", type=str, default="model_single_adam", help="子目录名")
parser.add_argument("--frequency", type=int, default=1, help="日志/评估频率（按 epoch）")

# 任务开关
parser.add_argument("--test",default=False, help="仅测试（不训练）")
parser.add_argument("--netdelay", action="store_true", default=True, help="是否训练/评估 net delay")
parser.add_argument("--celldelay", action="store_true", default=True, help="是否训练/评估 cell delay")
parser.add_argument("--norm", type=bool, required=True, help="是否使用归一化（保持原参数）")

# 数据划分
parser.add_argument("--train-graph-num", type=int, default=12, help="训练图数量")
parser.add_argument("--validate-graph-num", type=int, default=5, help="验证图数量")

# 优化器与调度
parser.add_argument("--lr", type=float, default=1e-3, help="初始学习率")
parser.add_argument("--min-lr", type=float, default=1e-8, help="CosineAnnealing 的最小学习率")
parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2")

# 断点
parser.add_argument("--resume", type=str, default=None, help="恢复训练的 checkpoint 路径（.tar）")

# 其他
parser.add_argument("--groundtruth", type=float, default=1.0, help="模型 forward 的 groundtruth 入口（兼容旧逻辑）")
parser.add_argument("--process-data", action="store_true", default=True, help="是否保存处理数据")
parser.add_argument('--opt',type=str, default='default',
                    help='using CircuitNet or not, "default" for paper test, "CircuitNet" for using CircuitNet')

# -----------------------------
# Train / Validate / Test
# -----------------------------
def train(model, dataloader, optimizer, epoch, scheduler, groundtruth, args):
    model.train()
    loss_net_sum = 0.0
    loss_cell_sum = 0.0
    loss_at_sum = 0.0

    for (g, ts), label in dataloader:
        optimizer.zero_grad()

        pred_net, pred_cell, pred_at, node_topo, cell_topo = model(g, ts, groundtruth=groundtruth)

        loss_net = 0.0
        loss_cell = 0.0

        if args.netdelay:
            loss_net = F.mse_loss(pred_net, g.ndata["n_net_delays"])
            loss_net_sum += float(loss_net.item())

        if args.celldelay:
            loss_cell = F.mse_loss(pred_cell, g.edges["cell_out"].data["e_cell_delays"])
            loss_cell_sum += float(loss_cell.item())
        else:
            # dgl 兼容（不参与梯度）
            loss_cell = torch.sum(pred_cell) * 0.0

        loss_at = F.mse_loss(pred_at, g.ndata["n_atslew"])
        loss_at_sum += float(loss_at.item())

        loss = loss_net + loss_cell + loss_at
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max(10 - 0.2 * epoch, 0.1))
        optimizer.step()

    # epoch 末尾调度（仅在未锁定时进行）
    if scheduler is not None:
        scheduler.step()

    if epoch % args.frequency == 0:
        with open(os.path.join(args.output_dir, args.checkpoint, "train_loss.txt"), "a") as f:
            f.write(
                f"{epoch},"
                f"{loss_net_sum / args.train_graph_num:.15f},"
                f"{loss_cell_sum / args.train_graph_num:.15f},"
                f"{loss_at_sum / args.train_graph_num:.15f}\n"
            )

    cur_lr = [pg["lr"] for pg in optimizer.param_groups]
    print(f"epoch:{epoch}: lr:{cur_lr} groundtruth:{groundtruth}")

    return (
        loss_net_sum / args.train_graph_num,
        loss_cell_sum / args.train_graph_num,
        loss_at_sum / args.train_graph_num,
    )


@torch.no_grad()
def validate(model, dataloader, epoch, args):
    model.eval()

    loss_net_sum = 0.0
    loss_cell_sum = 0.0
    loss_at_sum = 0.0
    loss_at_prop_sum = 0.0

    sample_r2 = []

    for (g, ts), label in dataloader:
        # groundtruth=True/False 两路
        pred_net, pred_cell, pred_at, _, _ = model(g, ts, groundtruth=1.0)
        pred_net_p, pred_cell_p, pred_at_p, _, _ = model(g, ts, groundtruth=0)

        true_at = g.ndata["n_atslew"][:, :4]
        pred_at4 = pred_at[:, :4]
        pred_at4_p = pred_at_p[:, :4]

        r2_pair = [-1, -1]
        if not pred_at4.isnan().any():
            r2_pair[0] = r2_score(true_at.cpu().numpy().reshape(-1), pred_at4.cpu().numpy().reshape(-1))
        if not pred_at4_p.isnan().any():
            r2_pair[1] = r2_score(true_at.cpu().numpy().reshape(-1), pred_at4_p.cpu().numpy().reshape(-1))
        sample_r2.append((r2_pair, g.num_nodes()))

        if args.netdelay:
            loss_net_sum += F.mse_loss(pred_net, g.ndata["n_net_delays"]).item()
        if args.celldelay:
            loss_cell_sum += F.mse_loss(pred_cell, g.edges["cell_out"].data["e_cell_delays"]).item()
        loss_at_sum += F.mse_loss(pred_at, g.ndata["n_atslew"]).item()
        loss_at_prop_sum += F.mse_loss(pred_at_p, g.ndata["n_atslew"]).item()

    # 加权节点数得到整体 R2
    nodes_tot = 0
    mean_r2 = 0.0
    mean_r2_prop = 0.0
    for r2_pair, n_nodes in sample_r2:
        mean_r2 += r2_pair[0] * n_nodes
        mean_r2_prop += r2_pair[1] * n_nodes
        nodes_tot += n_nodes
    mean_r2 /= float(nodes_tot)
    mean_r2_prop /= float(nodes_tot)

    logging.info(
        f"Epoch {epoch}, validate losses: "
        f"net {loss_net_sum / args.validate_graph_num:.6f}, "
        f"cell {loss_cell_sum / args.validate_graph_num:.6f}, "
        f"AT {loss_at_sum / args.validate_graph_num:.6f}, "
        f"AT_prop {loss_at_prop_sum / args.validate_graph_num:.6f}"
    )
    with open(os.path.join(args.output_dir, args.checkpoint, "validate_loss.txt"), "a") as f:
        f.write(
            f"{epoch},"
            f"{loss_net_sum / args.validate_graph_num:.15f},"
            f"{loss_cell_sum / args.validate_graph_num:.15f},"
            f"{loss_at_sum / args.validate_graph_num:.15f},"
            f"{loss_at_prop_sum / args.validate_graph_num:.15f}\n"
            f"{mean_r2:.15f},{mean_r2_prop}\n"
        )

    return mean_r2, mean_r2_prop


@torch.no_grad()
def test(model, dataloader, args):
    model.eval()
    args.norm = False
    sample_r2 = []
    for (g, ts), label in dataloader:
        pred_net, pred_cell, pred_at, _, _ = model(g, ts, groundtruth=True)
        pred_net_p, pred_cell_p, pred_at_p, _, _ = model(g, ts, groundtruth=False)

        r2 = [0, 0, 0, 0, 0, 0]
        true_at = g.ndata["n_atslew"][:, :4]
        true_net = g.ndata["n_net_delays_log"]
        true_cell = g.edges["cell_out"].data["e_cell_delays"]

        r2[0] = r2_score(true_at.cpu().numpy().reshape(-1), pred_at[:, :4].cpu().numpy().reshape(-1))
        r2[1] = r2_score(true_at.cpu().numpy().reshape(-1), pred_at_p[:, :4].cpu().numpy().reshape(-1))
        sample_r2.append((r2, g.num_nodes()))

        r2[2] = r2_score(true_net.cpu().numpy().reshape(-1), pred_net.cpu().numpy().reshape(-1))
        r2[3] = r2_score(true_net.cpu().numpy().reshape(-1), pred_net_p.cpu().numpy().reshape(-1))
        r2[4] = r2_score(true_cell.cpu().numpy().reshape(-1), pred_cell.cpu().numpy().reshape(-1))
        r2[5] = r2_score(true_cell.cpu().numpy().reshape(-1), pred_cell_p.cpu().numpy().reshape(-1))

        print(
            f"{label}: nodes:{g.num_nodes()}\n"
            f"\tAT R2:{r2[0]:.8f}    AT_prop R2:{r2[1]:.8f}\n"
            f"\tnet_delay R2:{r2[2]:.8f}   net_delay_prop R2:{r2[3]:.8f}\n"
            f"\tcell_delay R2:{r2[4]:.8f}  cell_delay_prop R2:{r2[5]:.8f}\n"
        )

    nodes_tot = 0
    mean_r2_prop = 0.0
    for r2_list, n_nodes in sample_r2:
        mean_r2_prop += r2_list[1] * n_nodes
        nodes_tot += n_nodes
    mean_r2_prop /= float(nodes_tot)
    return mean_r2_prop


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    args = parser.parse_args()

    # 输出目录
    ckpt_dir = os.path.join(args.output_dir, args.checkpoint)
    os.makedirs(ckpt_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(ckpt_dir, "TimingGCN.log"),
        format="%(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(args)

    # 设备 & 模型
    device = torch.device(args.device)
    model = TimingGCN().to(device)

    # 数据：从 data-root 收集 *.graph.bin
    dgl_graphs_path, labels = collect_graph_bins(args.data_root)
    if len(dgl_graphs_path) == 0:
        raise RuntimeError(f"在 {args.data_root} 下没有找到任何 *.graph.bin 文件")

    # DataLoaders
    dataloader_train, dataloader_validate = get_dataloder(
        dgl_graphs_path, labels, args.train_graph_num, args.validate_graph_num, args
    )

    # 单一 Adam 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )

    # 余弦退火调度（在 r2 没有达到阈值前启用）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - args.start_epoch),
        eta_min=args.min_lr,
    )

    # 断点恢复
    first_r2_reached_flag = False
    best_r2 = -1.0
    best_r2_prop = -1.0
    best_epoch = -1
    best_epoch_prop = -1

    if args.resume is not None and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and ckpt.get("lr_locked", False) is False:
            # 只有在未锁定 lr 的情况下才恢复 scheduler
            scheduler.load_state_dict(ckpt["scheduler"])
        args.start_epoch = ckpt.get("epoch", args.start_epoch) + 1
        best_r2 = ckpt.get("best_r2", best_r2)
        best_r2_prop = ckpt.get("best_r2_prop", best_r2_prop)
        first_r2_reached_flag = ckpt.get("lr_locked", False)
        if first_r2_reached_flag:
            set_optimizer_lr(optimizer, args.min_lr)  # 保持为 1e-8
            scheduler = None  # 不再使用调度器
        logging.info(f"Resumed from {args.resume} (start_epoch={args.start_epoch})")

    # 测试模式
    if args.test:
        r2_train = test(model, dataloader_train, args)
        r2_val = test(model, dataloader_validate, args)
        print(
            "mean R2_prop:",
            np.average(np.array([r2_train, r2_val]), weights=np.array([args.train_graph_num, args.validate_graph_num])),
        )
        raise SystemExit(0)

    print(f"Training TimingGCN on {args.device}")
    print(f"Saving logs & models to {ckpt_dir}")
    print(f"Start from epoch {args.start_epoch}")

    # groundtruth 兼容原逻辑（可由 validate 的表现调整）
    groundtruth = args.groundtruth
    groundtruth_scheduler = GrooundTruthScheduler(initial_groundtruth=args.groundtruth, start_epoch=args.start_epoch)

    for epoch in range(args.start_epoch, args.epochs):
        net_loss, cell_loss, at_loss = train(model, dataloader_train, optimizer, epoch, scheduler, groundtruth, args)

        if epoch % args.frequency == 0:
            r2, r2_prop = validate(model, dataloader_validate, epoch, args)

            # groundtruth 的自适应策略（沿用旧逻辑触发条件）
            if r2 > 0.8 or best_r2 > 0.9:
                if groundtruth > 0:
                    groundtruth = groundtruth_scheduler.step(net_loss, cell_loss, at_loss, epoch)

            # —— 学习率策略改动 —— #
            # 1) 在达到较高 r2 水平时略微调低 betas（保持原先意图）
            if r2 > 0.98 or r2_prop > 0.8:
                for pg in optimizer.param_groups:
                    pg["betas"] = (0.8, 0.999)

            # 2) 余弦退火直到 r2 第一次 > 0.88，随后将 lr 固定为 1e-8，并停用调度器
            if (not first_r2_reached_flag) and (r2 is not None) and (r2 > 0.88):
                first_r2_reached_flag = True
                set_optimizer_lr(optimizer, args.min_lr)  # 强制到 1e-8
                scheduler = None  # 之后不再 step
                logging.info(f"Epoch {epoch}: R2={r2:.6f} > 0.88, lock lr to {args.min_lr:g}")

            # 保存最优
            updated_best = False
            if r2 > best_r2:
                best_r2, best_epoch = r2, epoch
                updated_best = True
            if r2_prop > best_r2_prop:
                best_r2_prop, best_epoch_prop = r2_prop, epoch
                updated_best = True

            if updated_best:
                checkpoint_path = os.path.join(ckpt_dir, "checkpoint_best_r2.pth.tar")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": (scheduler.state_dict() if scheduler is not None else None),
                        "epoch": epoch,
                        "args": args,
                        "best_r2": best_r2,
                        "best_r2_prop": best_r2_prop,
                        "lr_locked": first_r2_reached_flag,  # 是否已锁定 1e-8
                    },
                    checkpoint_path,
                )

            logging.info(f"Epoch {epoch}, validate R2: {r2:.6f}, R2_prop: {r2_prop:.6f}")
            logging.info(f"Best R2 @ {best_epoch}: {best_r2:.6f}; Best R2_prop @ {best_epoch_prop}: {best_r2_prop:.6f}")

    print("Training done.")
