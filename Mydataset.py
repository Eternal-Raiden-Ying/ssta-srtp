import os.path

import dgl
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data_graph import *


class DGLGraphDataset(Dataset):
    def __init__(self, graphs_path, labels: list, opt='default', norm=False):
        """
        opt: default
            @param graphs_path: List[DGLGraph_path] - 一个包含多个 DGLGraph_path 的列表
            @param labels: List[str] - 每个图对应的标签
        opt: CircuitNet
            @param graphs_path: timing path
                                    |---- nodes
                                    |---- net_edges
                                    |---- pin_positions
            @labels: List[str]
        """
        self.graphs_path = graphs_path
        self.labels = labels
        self.norm = norm
        assert opt in ['default', 'CircuitNet'], f"invalid opt {opt}, expected 'default' or 'CircuitNet'"
        self.opt = opt

    def __len__(self):
        return len(self.graphs_path)

    def __getitem__(self, index):
        if self.opt == "default":
            g = dgl.load_graphs(self.graphs_path[index])[0][0].to("cuda",non_blocking=True)
            g_new, ts = graph_preprocess(g, normalize=self.norm)
            return (g_new, ts), self.labels[index]
        elif self.opt == "CircuitNet":
            # remain to be finished
            timing_path = self.graphs_path
            label = self.labels[index]
            net_edges = np.load(os.path.join(timing_path,"net_edges",f"{label}.npz"))['net_edges']
            nodes = np.load(os.path.join(timing_path,"nodes",f"{label}.npz"))['nodes']
            pin_positions = np.load(os.path.join(timing_path,"pin_positions",f"{label}.npz"),
                                    allow_pickle=True)['pin_positions'].item()

            # build a bi-direction graph for bi-direction message passing
            g = dgl.heterograph({
                ('node', 'net_out', 'node'): (net_edges[:, 0], net_edges[:, 1]),
                ('node', 'net_in', 'node'): (net_edges[:, 1], net_edges[:, 0]),
            })

            # assign net_delay to edge feature, which will be used as label in the following.
            g.edges['net_out'].data['net_delay'] = torch.tensor(net_edges[:, 2:]).type(torch.float32)

            # assign pin_positions to node feature.
            g.ndata['nf'] = torch.tensor(
                [pin_positions[nodes[i.item()].replace('\\', '')][0:4] for i in g.nodes()]).type(torch.float32)
            g.edges['net_out'].data['net_delays_log'] = (
                        torch.log(0.0001 + g.edges['net_out'].data['net_delay']) + 9.211)  # log(0.0001) ≈ -9.211


def collate_fn(batch):
    """
    批处理函数，将多个 DGLGraph 合并成一个 batch 图，并打包标签
    :param batch: List[(DGLGraph, label)]
    :return: (batched_graph, labels)
    """
    gts, labels = tuple(zip(*batch))
    graph, ts = gts[0]
    # batched_graph = dgl.batch(graphs)  # 合并成 batch
    # labels = torch.stack(labels)  # 转换成 tensor
    return (graph,ts), labels


def get_dataloder(graph_paths, labels, train_num, validate_num, args):
    assert len(graph_paths) == len(labels), ['graph paths not match labels']
    assert train_num + validate_num <= len(labels), ['sum of graph for training and validate beyond total graphs']
    index_list = [i for i in range(train_num+validate_num)]
    random.shuffle(index_list)
    train_paths = [graph_paths[i] for i in index_list[:train_num]]
    train_labels = [labels[i] for i in index_list[:train_num]]
    validate_paths = [graph_paths[i] for i in index_list[-validate_num:]]
    validate_labels = [labels[i] for i in index_list[-validate_num:]]

    data_loader_train = torch.utils.data.DataLoader(
        dataset=DGLGraphDataset(graphs_path=train_paths, labels=train_labels, opt=args.opt, norm=args.norm),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_fn
    )

    data_loader_validate = torch.utils.data.DataLoader(
        dataset=DGLGraphDataset(graphs_path=validate_paths, labels=validate_labels, opt=args.opt, norm=args.norm),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=collate_fn
    )
    return data_loader_train, data_loader_validate

def get_dataloader_from_circuitNet(timing_path, labels, train_num, validate_num, args):
    """"
        timing path
            |---- nodes
            |---- net_edges
            |---- pin_positions
    """
    # remain to be finished
    assert train_num + validate_num <= len(labels), ['sum of graph for training and validate beyond total graphs']
    assert os.path.exists(timing_path), "timing path not exist"
    index_list = [i for i in range(train_num + validate_num)]
    random.shuffle(index_list)
    train_labels = [labels[i] for i in index_list[:train_num]]
    validate_labels = [labels[i] for i in index_list[-validate_num:]]

    data_loader_train = torch.utils.data.DataLoader(
        dataset=DGLGraphDataset(graphs_path=timing_path, labels=train_labels, opt=args.opt),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_fn
    )

    data_loader_validate = torch.utils.data.DataLoader(
        dataset=DGLGraphDataset(graphs_path=timing_path, labels=validate_labels, opt=args.opt),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=collate_fn
    )
    return data_loader_train, data_loader_validate

