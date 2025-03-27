import torch
import dgl
import numpy as np
from data_graph import graph_preprocess

net_edges = np.load(r"E:\BaiduNetdiskDownload\CircuitNet-N28\timing_features\net_edges\2-RISCY-a-1-c2-u0.7-m1-p2-f0.npz")['net_edges']
nodes = np.load(r"E:\BaiduNetdiskDownload\CircuitNet-N28\timing_features\nodes\2-RISCY-a-1-c2-u0.7-m1-p2-f0.npz")['nodes']
pin_positions = np.load(r"E:\BaiduNetdiskDownload\CircuitNet-N28\timing_features\pin_positions\2-RISCY-a-1-c2-u0.7-m1-p2-f0.npz", allow_pickle=True)['pin_positions'].item()

# build a bi-direction graph for bi-direction message passing
g = dgl.heterograph({
('node', 'net_out', 'node'): (net_edges[:,0], net_edges[:,1]),
('node', 'net_in', 'node'): (net_edges[:,1], net_edges[:,0]),
})

# assign net_delay to edge feature, which will be used as label in the following.
g.edges['net_out'].data['net_delay'] = torch.tensor(net_edges[:,2:]).type(torch.float32)

# assign pin_positions to node feature.
g.ndata['nf'] = torch.tensor([pin_positions[nodes[i.item()].replace('\\','')][0:4] for i in g.nodes()]).type(torch.float32)
g.edges['net_out'].data['net_delays_log'] = (torch.log(0.0001 + g.edges['net_out'].data['net_delay']) + 9.211) # log(0.0001) ≈ -9.211

graph_preprocess(g)

for (g, ts), label in dataloader:
    pred_net_delays, pred_cell_delays, pred_atslew = model(g, ts, groundtruth=args.groundtruth)
    loss_net_delays, loss_cell_delays = 0, 0

    loss_net_delays = F.mse_loss(pred_net_delays, g.ndata['n_net_delays_log'])

    loss_cell_delays = F.mse_loss(pred_cell_delays, g.edges['cell_out'].data['e_cell_delays'])

    loss_ats = F.mse_loss(pred_atslew, g.ndata['n_atslew'])

    (loss_net_delays + loss_cell_delays + loss_ats).backward()