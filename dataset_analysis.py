import os.path

import torch
import dgl
import numpy as np
import argparse
from model import TimingGCN,SignalProp,SignalPropAttn
from Mydataset import get_dataloder
from sklearn.metrics import r2_score
from data_graph import dgl_graphs_path, labels
from utils import log_cosh_loss, mae_loss
import json
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='TimingPredict-GNN')
parser.add_argument('--batch-size', type=int, default=1, help='batch size')
parser.add_argument('--num-workers',type=int, default=0, help='number of workers')
parser.add_argument('--pin-mem', type=bool, default=False, help='pin memory')
parser.add_argument('--opt',type=str, default='default')
parser.add_argument('--output_dir',type=str, default='numerical_analysis')

if __name__ == "__main__":
    args = parser.parse_args()
    checkpoint_path = r"E:\ssta_gnn\TimingPredict-master\TimingPredict-master\checkpoints\08_atcd_specul\15799.pth"
    model = TimingGCN()
    model2 = TimingGCN()
    signal_prop = model.prop
    checkpoint = torch.load(checkpoint_path,map_location='cpu')
    model.load_state_dict(checkpoint,strict=True)
    model2.load_state_dict(checkpoint,strict=True)
    assert torch.cuda.is_available(), 'cuda is not available'
    model.cuda()
    model2.cuda()
    with torch.no_grad():
        for name, parameter in model2.named_parameters():
            if 'weight' in name:
                max_value = torch.max(parameter)
                parameter.view(-1)[torch.argmax(parameter)]=0
                print(name,f"max_value {max_value} difference norm:", torch.norm(parameter - model.state_dict()[name]))

        # data_loader, _ = get_dataloder(dgl_graphs_path,labels,15,0, args=args)
        # data_loader, _ = get_dataloder(['data/8_rat/spm.graph.bin',], ['spm',], 1, 0, args=args)
        data_loader, _ = get_dataloder(['data/8_rat/picorv32a.graph.bin',], ['blabla',], 1, 0, args=args)

        for (graph, tensors), label in data_loader:
            if isinstance(label, tuple) and len(label) == 1:
                label = label[0]
            assert isinstance(label, str), [f'TypeError, received label with type {type(label)}, {label}']
            print(label, "  ", graph.num_nodes())
            for key in graph.ndata:
                print(f"{label} {key}:",
                      graph.ndata[key].shape,torch.max(graph.ndata[key],dim=0).values,
                      graph.ndata[key].shape,torch.min(graph.ndata[key],dim=0).values)
            for key in graph.edges['net_out'].data:
                print(f"{label}  net_out {key}:",
                      graph.edges['net_out'].data[key].shape,torch.max(graph.edges['net_out'].data[key],dim=0).values,
                      graph.edges['net_out'].data[key].shape,torch.min(graph.edges['net_out'].data[key],dim=0).values)
            for key in graph.edges['cell_out'].data:
                print(f"{label}  cell_out {key}:",
                      graph.edges['cell_out'].data[key].shape,torch.max(graph.edges['cell_out'].data[key],dim=0).values,
                      graph.edges['cell_out'].data[key].shape,torch.min(graph.edges['cell_out'].data[key],dim=0).values)
            # pred_net_delays, pred_cell_delays, pred_atslew, node_topo, cell_topo = model(graph, tensors, False)
            # pred_net_delays2, pred_cell_delays2, pred_atslew2, node_topo2, cell_topo2 = model2(graph, tensors, False)
            # truth = graph.ndata['n_atslew']
            # r2 = r2_score(truth.cpu().numpy().reshape(-1), pred_atslew.cpu().numpy().reshape(-1))
            # print(f"{label}: raw model r2: {r2:.15f}")
            # r2 = r2_score(truth.cpu().numpy().reshape(-1), pred_atslew2.cpu().numpy().reshape(-1))
            # print(f"{label}: dropout model r2: {r2:.15f}")
            print()



