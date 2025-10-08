import torch
import dgl
import random
import time


random.seed(8026728)

# available_data = ('blabla usb_cdc_core BM64 jpeg_encoder salsa20 '
#                            'usbf_device aes128 wbqspiflash aes192 cic_decimator '
#                            'xtea aes256 des spm y_huff aes_cipher picorv32a synth_ram '
#                            'zipdiv genericfir usb').split()

available_data = ('blabla usb_cdc_core BM64 salsa20 '
                  'usbf_device wbqspiflash cic_decimator '
                  'xtea des spm y_huff aes_cipher picorv32a synth_ram '
                  'zipdiv genericfir usb').split()  # small version

# available_data = ('usb_cdc_core BM64 salsa20 '
#                   'usbf_device wbqspiflash cic_decimator '
#                   'xtea des spm y_huff aes_cipher picorv32a synth_ram '
#                   'zipdiv genericfir usb').split()  # blabla always cause nan

"""
    num nodes: number of total nodes
    num edges: number of total cell_out edge and net_out edge  (physical edges)

    blabla:         num_nodes:55568     num_edges:75542
    usb_cdc_core:   num_nodes:7406      num_edges:10069
    BM64:           num_nodes:38458     num_edges:53177
    jpeg_encoder:   num_nodes:238216    num_edges:344697
    salsa20:        num_nodes:78486     num_edges:110632
    usbf_device:    num_nodes:66345     num_edges:88467
    aes128:         num_nodes:211045    num_edges:287454
    wbqspiflash:    num_nodes:9672      num_edges:13252
    aes192:         num_nodes:234211    num_edges:318260
    cic_decimator:  num_nodes:3131      num_edges:4334
    xtea:           num_nodes:10213     num_edges:14033
    aes256:         num_nodes:290955    num_edges:396676
    des:            num_nodes:60541     num_edges:86323
    spm:            num_nodes:1121      num_edges:1465
    y_huff:         num_nodes:48216     num_edges:64301
    aes_cipher:     num_nodes:59777     num_edges:84082
    picorv32a:      num_nodes:58676     num_edges:83255
    synth_ram:      num_nodes:25910     num_edges:35806
    zipdiv:         num_nodes:4398      num_edges:6015
    genericfir:     num_nodes:38827     num_edges:53858
    usb:            num_nodes:3361      num_edges:4595
"""

labels = available_data
dgl_graphs_path = ['data/8_rat/{}.graph.bin'.format(k) for k in labels]


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


def gen_homobigraph_with_features(g_hetero):
    # for DeepGCNII baseline
    na, nb = g_hetero.edges(etype='net_out', form='uv')
    ca, cb = g_hetero.edges(etype='cell_out', form='uv')
    ne = torch.cat([torch.tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]]).expand(len(na), 10).cuda(),
                    g_hetero.edges['net_out'].data['ef']], dim=1)
    ce = g_hetero.edges['cell_out'].data['ef'][:, 120:512].reshape(len(ca), 2*4, 49)
    ce = torch.cat([torch.tensor([[1., 0.]]).expand(len(ca), 2).cuda(),
                    torch.mean(ce, dim=2),
                    torch.zeros(len(ca), 2).cuda()], dim=1)
    g = dgl.graph((torch.cat([na, ca, nb, cb]), torch.cat([nb, cb, na, ca])))
    g.ndata['nf'] = g_hetero.ndata['nf']
    g.ndata['n_atslew'] = g_hetero.ndata['n_atslew']
    g.edata['ef'] = torch.cat([ne, ce, -ne, -ce])
    return g


def get_data(key:str, device:str):
    if 'cuda' in device and not torch.cuda.is_available():
        print(f"{device} is not available, using cpu to load data from {key}")
        device = "cpu"
    device = torch.device(device if torch.cuda.is_available() else "cpu")



def graph_preprocess(g, normalize=False):
    # # v11 ours
    g.ndata['n_net_delays_log'] = torch.log(0.0001 + g.ndata['n_net_delays']) + 7.6
    invalid_nodes = torch.abs(g.ndata['n_ats']) > 1e20  # ignore all uninitialized stray pins
    g.ndata['n_ats'][invalid_nodes] = 0
    g.ndata['n_slews'][invalid_nodes] = 0
    g.ndata['n_atslew'] = torch.cat([
        g.ndata['n_ats'],
        torch.log(0.00001 + g.ndata['n_slews']) + 11.6
    ], dim=1)  # g.ndata['n_ats'].shape -> Nodes, Features(EL/RF)
    g.edges['cell_out'].data['ef'] = g.edges['cell_out'].data['ef'].type(torch.float32)
    g.edges['cell_out'].data['e_cell_delays'] = g.edges['cell_out'].data['e_cell_delays'].type(torch.float32)

    # new
    if normalize:
        nodetype, nodepos, nodecap = torch.split(g.ndata['nf'],[2,4,4],dim=1)
        g.ndata['nf'] = torch.cat([nodetype, nodepos/400,torch.log(nodecap*10000+1)],dim=1)
        query, tables = torch.split(g.edges['cell_out'].data['ef'],[120,392],dim=1)
        is_table_valid, x_axis, y_axis = torch.split(query.reshape(-1,8,15), [1,7,7],dim=2)
        query = torch.cat([is_table_valid, torch.log(0.00001+x_axis)+11.6, torch.log(y_axis*10000+1)], dim=2).reshape(-1,120)
        g.edges['cell_out'].data['ef'] = torch.cat([query,tables], dim=1)

        g.edges['net_out'].data['ef'] = g.edges['net_out'].data['ef']/400
        g.edges['net_in'].data['ef'] = g.edges['net_in'].data['ef']/400

    ########################################################
    # # raw, baseline
    else:
        g.ndata['n_net_delays_log'] = torch.log(0.0001 + g.ndata['n_net_delays']) + 7.6
        invalid_nodes = torch.abs(g.ndata['n_ats']) > 1e20  # ignore all uninitialized stray pins
        g.ndata['n_ats'][invalid_nodes] = 0
        g.ndata['n_slews'][invalid_nodes] = 0
        g.ndata['n_atslew'] = torch.cat([
            g.ndata['n_ats'],
            torch.log(0.0001 + g.ndata['n_slews']) + 3
        ], dim=1)  # g.ndata['n_ats'].shape -> Nodes, Features(EL/RF)
        g.edges['cell_out'].data['ef'] = g.edges['cell_out'].data['ef'].type(torch.float32)
        g.edges['cell_out'].data['e_cell_delays'] = g.edges['cell_out'].data['e_cell_delays'].type(torch.float32)

    ########################################################
    #common
    topo, topo_time = gen_topo(g)

    # g.ndata['nf']: node features (is_I/O_pin, is_fanout, distance*4, capacitance*4)
    ts = {'input_nodes': (g.ndata['nf'][:, 1] < 0.5).nonzero().flatten().type(torch.int32),  # fp64 -> fp32
          'output_nodes': (g.ndata['nf'][:, 1] > 0.5).nonzero().flatten().type(torch.int32),  # nonzero 找到非零元素的索引
          'output_nodes_nonpi': torch.logical_and(g.ndata['nf'][:, 1] > 0.5,
                                                  g.ndata['nf'][:, 0] < 0.5).nonzero().flatten().type(torch.int32),
          'pi_nodes': torch.logical_and(g.ndata['nf'][:, 1] > 0.5, g.ndata['nf'][:, 0] > 0.5).nonzero().flatten().type(
              torch.int32),
          'po_nodes': torch.logical_and(g.ndata['nf'][:, 1] < 0.5, g.ndata['nf'][:, 0] > 0.5).nonzero().flatten().type(
              torch.int32),
          'endpoints': (g.ndata['n_is_timing_endpt'] > 0.5).nonzero().flatten().type(torch.long),
          'topo': topo,
          'topo_time': topo_time}
    return g, ts


# train_data_keys = random.sample(available_data, 14)
# data = dict()
#
# for k in available_data:
#     g = dgl.load_graphs('data/8_rat/{}.graph.bin'.format(k))[0][0].to('cuda')
#     g.ndata['n_net_delays_log'] = torch.log(0.0001 + g.ndata['n_net_delays']) + 7.6
#     invalid_nodes = torch.abs(g.ndata['n_ats']) > 1e20   # ignore all uninitialized stray pins
#     g.ndata['n_ats'][invalid_nodes] = 0
#     g.ndata['n_slews'][invalid_nodes] = 0
#     g.ndata['n_atslew'] = torch.cat([
#         g.ndata['n_ats'],
#         torch.log(0.0001 + g.ndata['n_slews']) + 3
#     ], dim=1)  # g.ndata['n_ats'].shape -> Nodes, Features(EL/RF)
#     g.edges['cell_out'].data['ef'] = g.edges['cell_out'].data['ef'].type(torch.float32)
#     g.edges['cell_out'].data['e_cell_delays'] = g.edges['cell_out'].data['e_cell_delays'].type(torch.float32)
#     topo, topo_time = gen_topo(g)
#
#     # g.ndata['nf']: node features (is_I/O_pin, is_fanout, distance*4, capacitance*4)
#     ts = {'input_nodes': (g.ndata['nf'][:, 1] < 0.5).nonzero().flatten().type(torch.int32),  # fp64 -> fp32
#           'output_nodes': (g.ndata['nf'][:, 1] > 0.5).nonzero().flatten().type(torch.int32),  # nonzero 找到非零元素的索引
#           'output_nodes_nonpi': torch.logical_and(g.ndata['nf'][:, 1] > 0.5, g.ndata['nf'][:, 0] < 0.5).nonzero().flatten().type(torch.int32),
#           'pi_nodes': torch.logical_and(g.ndata['nf'][:, 1] > 0.5, g.ndata['nf'][:, 0] > 0.5).nonzero().flatten().type(torch.int32),
#           'po_nodes': torch.logical_and(g.ndata['nf'][:, 1] < 0.5, g.ndata['nf'][:, 0] > 0.5).nonzero().flatten().type(torch.int32),
#           'endpoints': (g.ndata['n_is_timing_endpt'] > 0.5).nonzero().flatten().type(torch.long),
#           'topo': topo,
#           'topo_time': topo_time}
#     data[k] = g, ts
#
# data_train = {k: t for k, t in data.items() if k in train_data_keys}
# data_test = {k: t for k, t in data.items() if k not in train_data_keys}
# print()




# if __name__ == '__main__':
#     # print('Graph statistics: (total {} graphs)'.format(len(data)))
#     # print('{:15} {:>10} {:>10}'.format('NAME', '#NODES', '#EDGES'))
#     # for k, (g, ts) in data.items():
#     #     print('{:15} {:>10} {:>10}'.format(k, g.num_nodes(), g.num_edges()))
#     for dic in [data_train, data_test]:
#         for k, (g, ts) in dic.items():
#             print('\\texttt{{{}}},{},{},{},{},{},{}'.format(k.replace('_', '\_'), g.num_nodes(), g.num_edges('net_out'), g.num_edges('cell_out'), len(ts['topo']), len(ts['po_nodes']), len(ts['endpoints'])))
