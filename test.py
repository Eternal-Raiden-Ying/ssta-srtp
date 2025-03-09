import torch
import numpy as np
import dgl
import torch.nn.functional as F
import random
import pdb
import time
import argparse
import os
from sklearn.metrics import r2_score
import tee
from model import TimingGCN


def gen_topo(g_hetero):
    torch.cuda.synchronize()
    time_s = time.time()
    na, nb = g_hetero.edges(etype='net_out', form='uv')
    ca, cb = g_hetero.edges(etype='cell_out', form='uv')
    g = dgl.graph((torch.cat([na, ca]).cpu(), torch.cat([nb, cb]).cpu()))
    topo = dgl.topological_nodes_generator(g)
    ret = [t.cuda() for t in topo]
    # 拓扑排序
    # refer:
    # https://www.dgl.ai/dgl_docs/en/1.1.x/generated/dgl.topological_nodes_generator.html#dgl.topological_nodes_generator
    torch.cuda.synchronize()
    time_e = time.time()
    return ret, time_e - time_s


# available_data = ('blabla usb_cdc_core BM64 jpeg_encoder salsa20 '
#                   'usbf_device aes128 wbqspiflash aes192 cic_decimator '
#                   'xtea aes256 des spm y_huff aes_cipher picorv32a synth_ram '
#                   'zipdiv genericfir usb').split()
available_data = ['blabla',]

data = {}
for k in available_data:
    g = dgl.load_graphs('data/8_rat/{}.graph.bin'.format(k))[0][0].to('cuda')
    g.ndata['n_net_delays_log'] = torch.log(0.0001 + g.ndata['n_net_delays']) + 7.6
    invalid_nodes = torch.abs(g.ndata['n_ats']) > 1e20   # ignore all uninitialized stray pins
    g.ndata['n_ats'][invalid_nodes] = 0
    g.ndata['n_slews'][invalid_nodes] = 0
    g.ndata['n_atslew'] = torch.cat([
        g.ndata['n_ats'],
        torch.log(0.0001 + g.ndata['n_slews']) + 3
    ], dim=1)  # g.ndata['n_ats'].shape -> Nodes, Features(EL/RF)
    g.edges['cell_out'].data['ef'] = g.edges['cell_out'].data['ef'].type(torch.float32)
    g.edges['cell_out'].data['e_cell_delays'] = g.edges['cell_out'].data['e_cell_delays'].type(torch.float32)
    topo, topo_time = gen_topo(g)

    # g.ndata['nf']: node features (is_I/O_pin, is_fanout, distance*4, capacitance*4)
    ts = {'input_nodes': (g.ndata['nf'][:, 1] < 0.5).nonzero().flatten().type(torch.int32),  # fp64 -> fp32
          'output_nodes': (g.ndata['nf'][:, 1] > 0.5).nonzero().flatten().type(torch.int32),  # nonzero 找到非零元素的索引
          'output_nodes_nonpi': torch.logical_and(g.ndata['nf'][:, 1] > 0.5, g.ndata['nf'][:, 0] < 0.5).nonzero().flatten().type(torch.int32),
          'pi_nodes': torch.logical_and(g.ndata['nf'][:, 1] > 0.5, g.ndata['nf'][:, 0] > 0.5).nonzero().flatten().type(torch.int32),
          'po_nodes': torch.logical_and(g.ndata['nf'][:, 1] < 0.5, g.ndata['nf'][:, 0] > 0.5).nonzero().flatten().type(torch.int32),
          'endpoints': (g.ndata['n_is_timing_endpt'] > 0.5).nonzero().flatten().type(torch.long),
          'topo': topo,
          'topo_time': topo_time}
    data[k] = g, ts

data_train = data

model = TimingGCN().cuda()
checkpoint_path = r"E:\ssta_gnn\TimingPredict-master\TimingPredict-master\checkpoints\08_atcd_specul\15799.pth"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint, strict=True)

(graph, tensors) = data_train['blabla']
start = time.time()
model.train()
predict = model(graph, tensors, groundtruth=False)
end = time.time()
print(end-start)