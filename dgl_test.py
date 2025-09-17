# import torch
# import dgl
# import numpy as np
# from data_graph import graph_preprocess
#
# net_edges = np.load(r"E:\BaiduNetdiskDownload\CircuitNet-N28\timing_features\net_edges\2-RISCY-a-1-c2-u0.7-m1-p2-f0.npz")['net_edges']
# nodes = np.load(r"E:\BaiduNetdiskDownload\CircuitNet-N28\timing_features\nodes\2-RISCY-a-1-c2-u0.7-m1-p2-f0.npz")['nodes']
# pin_positions = np.load(r"E:\BaiduNetdiskDownload\CircuitNet-N28\timing_features\pin_positions\2-RISCY-a-1-c2-u0.7-m1-p2-f0.npz", allow_pickle=True)['pin_positions'].item()
#
# # build a bi-direction graph for bi-direction message passing
# g = dgl.heterograph({
# ('node', 'net_out', 'node'): (net_edges[:,0], net_edges[:,1]),
# ('node', 'net_in', 'node'): (net_edges[:,1], net_edges[:,0]),
# })
#
# # assign net_delay to edge feature, which will be used as label in the following.
# g.edges['net_out'].data['net_delay'] = torch.tensor(net_edges[:,2:]).type(torch.float32)
#
# # assign pin_positions to node feature.
# g.ndata['nf'] = torch.tensor([pin_positions[nodes[i.item()].replace('\\','')][0:4] for i in g.nodes()]).type(torch.float32)
# g.edges['net_out'].data['net_delays_log'] = (torch.log(0.0001 + g.edges['net_out'].data['net_delay']) + 9.211) # log(0.0001) ≈ -9.211
#
# graph_preprocess(g)
#
# for (g, ts), label in dataloader:
#     pred_net_delays, pred_cell_delays, pred_atslew = model(g, ts, groundtruth=args.groundtruth)
#     loss_net_delays, loss_cell_delays = 0, 0
#
#     loss_net_delays = F.mse_loss(pred_net_delays, g.ndata['n_net_delays_log'])
#
#     loss_cell_delays = F.mse_loss(pred_cell_delays, g.edges['cell_out'].data['e_cell_delays'])
#
#     loss_ats = F.mse_loss(pred_atslew, g.ndata['n_atslew'])
#
#     (loss_net_delays + loss_cell_delays + loss_ats).backward()
#

import torch
import dgl
from dgl import function as fn
import torch.nn.functional as F
import time

# 定义边的起点（src）和终点（dst）
edges = {
    ('node', 'net', 'node'): (torch.tensor([1, 2, 3, 7, 8, 11])-1,
                              torch.tensor([4, 5, 6, 9, 10, 12])-1),
    ('node', 'cell', 'node'): (torch.tensor([4, 5, 6, 9, 10])-1,
                               torch.tensor([7, 7, 8, 11, 11])-1)
}

# 创建异构图
g = dgl.heterograph(edges)

lin = torch.nn.Linear(3,1,bias=False)
torch.nn.init.constant_(lin.weight, 1.0)
lin2 = torch.nn.Linear(3,1,bias=False)
torch.nn.init.constant_(lin2.weight,-1.0)
def message_func(edges):
    x = torch.cat([edges.src['nf'].reshape(-1,1), edges.dst['nf'].reshape(-1,1), edges.data['ef'].reshape(-1,1)], dim=1)
    # x=edges.data['ef'].reshape(-1,1)
    x=lin(x)
    return {'new_nf':x}

def message_func2(edges):
    x = torch.cat([edges.src['nf'].reshape(-1,1), edges.dst['nf'].reshape(-1,1), edges.data['ef'].reshape(-1,1)], dim=1)
    # x=edges.data['ef'].reshape(-1,1)
    x=lin2(x)
    return {'new_nf':x}

def gen_topo(g_hetero):
    time_s = time.time()
    na, nb = g_hetero.edges(etype='net', form='uv')
    ca, cb = g_hetero.edges(etype='cell', form='uv')
    g = dgl.graph((torch.cat([na, ca]).cpu(), torch.cat([nb, cb]).cpu()))
    topo = dgl.topological_nodes_generator(g)
    ret = [t.cpu() for t in topo]
    # 拓扑排序
    # refer:
    # https://www.dgl.ai/dgl_docs/en/1.1.x/generated/dgl.topological_nodes_generator.html#dgl.topological_nodes_generator
    torch.cuda.synchronize()
    time_e = time.time()
    return ret, time_e - time_s

g.ndata['nf'] = torch.arange(1,13).to(torch.float32).reshape(-1,1)
g.edges['net'].data['ef'] = (torch.arange(1,7)*10).to(torch.float32).reshape(-1,1)
g.edges['cell'].data['ef'] = (torch.arange(1,6)*100).to(torch.float32).reshape(-1,1)
# g.update_all(message_func,fn.sum('new_nf','nf2'),etype='cell')
# print(g.ndata['nf2'])
# s = g.ndata['nf2'].shape
# loss = F.mse_loss(g.ndata['nf2'],torch.zeros(s))
topo,_ = gen_topo(g)
for i in range(1,4):
    if i%2==1:
        edges = g.in_edges(topo[i],etype='net')
        g.send_and_recv(edges,message_func=message_func,reduce_func=fn.sum('new_nf','nf'),etype='net')
    else:
        edges = g.in_edges(topo[i], etype='cell')
        g.send_and_recv(edges, message_func=message_func2, reduce_func=fn.sum('new_nf', 'nf'), etype='cell')
# g.send_and_recv(edges=g.in_edges(torch.tensor([3,4,5],dtype=torch.int32),etype='net'),message_func=message_func,reduce_func=fn.sum('new_nf','nf'),etype='net')
# g.send_and_recv(edges=g.in_edges(torch.tensor([6,7],dtype=torch.int32),etype='cell'),message_func=message_func2,reduce_func=fn.sum('new_nf','nf'),etype='cell')
# g.send_and_recv(edges=g.in_edges(torch.tensor([8,9],dtype=torch.int32),etype='net'),message_func=message_func,reduce_func=fn.sum('new_nf','nf'),etype='net')
# g.send_and_recv(edges=g.in_edges(torch.tensor([10],dtype=torch.int32),etype='cell'),message_func=message_func2,reduce_func=fn.sum('new_nf','nf'),etype='cell')
# g.send_and_recv(edges=g.in_edges(torch.tensor([11],dtype=torch.int32),etype='net'),message_func=message_func,reduce_func=fn.sum('new_nf','nf'),etype='net')
loss = F.mse_loss(g.ndata['nf'][9],torch.zeros(1))
print(g.ndata['nf'])
loss.backward()
g.ndata['nf2'].grad()
