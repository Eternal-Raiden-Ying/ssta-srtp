import torch
import torch.nn.functional as F
import dgl
import dgl.function as fn
import functools
import pdb


class MLP(torch.nn.Module):
    """
    sizes: (A,B,C): e.g.
        -> MLP:
            nn.Linear(A,B) - nn.Linear(B,C)
    """
    def __init__(self, *sizes, batchnorm=False, dropout=False):
        super().__init__()
        fcs = []
        for i in range(1, len(sizes)):
            fcs.append(torch.nn.Linear(sizes[i - 1], sizes[i]))
            if i < len(sizes) - 1:
                fcs.append(torch.nn.LeakyReLU(negative_slope=0.2))
                if dropout: fcs.append(torch.nn.Dropout(p=0.2))  # p possibility of dropout when training
                if batchnorm: fcs.append(torch.nn.BatchNorm1d(sizes[i]))  # 批归一化 1d use for (N,C,L)
        self.layers = torch.nn.Sequential(*fcs)

    def forward(self, x):
        return self.layers(x)


class NetConv(torch.nn.Module):
    """
    edges.type:: net_in, net_out, cell_out
           The net_in edges are just reversed net_out edges
           reference: https://github.com/TimingPredict/Dataset/issues/1
        net_in, net_out:
            features:
                (2x): distance
        cell_out:
            features:
                (2*4*(1+7+7)x): [E/L]* cell_{rise, fall}, {rise, fall}_transition {is_valid, xindex, yindex}
                (2*4*49x):      [E/L]* cell_{rise, fall}, {rise, fall}_transition values
                (4x): _task_    cell delay annotations (EL/RF)

    nodes.type:: node
        features:
            (1x): is I/O pin
            (1x): is fanout
            (4x): position
            (4x): capacitance(EL/RF)
            below are _task_
            (4x): net delay for fanin pin
            (4x): AT
            (4x): slew
            (1x): is timing endpoint (has constraint)
            (4x): RT

            (4x): net delay log
            (8x): AT & slew,  (n_atslew)
    """
    def __init__(self, in_nf, in_ef, out_nf, h1=32, h2=32):
        super().__init__()
        self.in_nf = in_nf  # input node features
        self.in_ef = in_ef  # input edge features
        self.out_nf = out_nf
        self.h1 = h1  # h1 and h2 stand for dimension of sum and max
        self.h2 = h2
        
        self.MLP_msg_i2o = MLP(self.in_nf * 2 + self.in_ef, 64, 64, 64, 1 + self.h1 + self.h2)
        self.MLP_reduce_o = MLP(self.in_nf + self.h1 + self.h2, 64, 64, 64, self.out_nf)
        self.MLP_msg_o2i = MLP(self.in_nf * 2 + self.in_ef, 64, 64, 64, 64, self.out_nf)

    def edge_msg_i(self, edges):
        # x.shape , nodes_num , features
        # edges type: net out, fanout node -> fanin node
        x = torch.cat([edges.src['nf'], edges.dst['nf'], edges.data['ef']], dim=1)
        x = self.MLP_msg_o2i(x)
        return {'efi': x}

    def edge_msg_o(self, edges):
        # edges type: net in, fanin node -> fanout node (other cell)
        x = torch.cat([edges.src['nf'], edges.dst['nf'], edges.data['ef']], dim=1)
        x = self.MLP_msg_i2o(x)
        k, f1, f2 = torch.split(x, [1, self.h1, self.h2], dim=1)
        k = torch.sigmoid(k)  # TODO:take a look
        # using k to simulate attention scheme
        return {'efo1': f1 * k, 'efo2': f2 * k}

    def node_reduce_o(self, nodes):
        # fanout nodes
        x = torch.cat([nodes.data['nf'], nodes.data['nfo1'], nodes.data['nfo2']], dim=1)
        x = self.MLP_reduce_o(x)
        return {'new_nf': x}
        
    def forward(self, g, ts, nf):
        with g.local_scope():
            g.ndata['nf'] = nf
            # input nodes of cell
            g.update_all(self.edge_msg_i, fn.sum('efi', 'new_nf'), etype='net_out')
            # update graph.edges['net_out']
            # Mes(edges['net_out'].src, edges['net_out'].dst, edges['net_out'].data)
            #   --MLP--> edges['net_out'].mailbox['efi']
            #   --fn.sum--> edges['net_out'].dst.data['new_nf']
            # actually all nodes implemented operation fn.sum, nodes with no 'net_out' in-edge gets torch.zero in 'new_nf'
                # TODO: could this be replaced by g.pull()?

            # output nodes
            g.apply_edges(self.edge_msg_o, etype='net_in')
            # update features of edges['net_in']
            # edges['net_in'].dst, which is output nodes of cells, thus got info of output capacitance
            # but only save channel for sum and max though?????

            g.update_all(fn.copy_e('efo1', 'efo1'), fn.sum('efo1', 'nfo1'), etype='net_in')
            g.update_all(fn.copy_e('efo2', 'efo2'), fn.max('efo2', 'nfo2'), etype='net_in')
            g.apply_nodes(self.node_reduce_o, ts['output_nodes'])  # apply to all fanout nodes

            return g.ndata['new_nf']


class SignalProp(torch.nn.Module):
    def __init__(self, in_nf, in_cell_num_luts, in_cell_lut_sz, out_nf, out_cef, h1=32, h2=32, lut_dup=4):
        super().__init__()
        self.in_nf = in_nf  # 10+16
        self.in_cell_num_luts = in_cell_num_luts  # 8
        self.in_cell_lut_sz = in_cell_lut_sz  # 7
        self.out_nf = out_nf  # 8
        self.out_cef = out_cef  # 4
        self.h1 = h1
        self.h2 = h2
        self.lut_dup = lut_dup  # ??
        
        self.MLP_netprop = MLP(self.out_nf + 2 * self.in_nf, 64, 64, 64, 64, self.out_nf)
        self.MLP_lut_query = MLP(self.out_nf + 2 * self.in_nf, 64, 64, 64, self.in_cell_num_luts * lut_dup * 2)
        self.MLP_lut_attention = MLP(1 + 2 + self.in_cell_lut_sz * 2, 64, 64, 64, self.in_cell_lut_sz * 2)
        self.MLP_cellarc_msg = MLP(self.out_nf + 2 * self.in_nf + self.in_cell_num_luts * self.lut_dup, 64, 64, 64, 1 + self.h1 + self.h2 + self.out_cef)
        self.MLP_cellreduce = MLP(self.in_nf + self.h1 + self.h2, 64, 64, 64, self.out_nf)

    def edge_msg_net(self, edges, groundtruth=False):
        # edges type: net_out, fanout node -> fanin node
        if groundtruth:
            # for training
            last_nf = edges.src['n_atslew']  # AT and slew of fanout node of last cell
        else:
            # for validating
            last_nf = edges.src['new_nf']
        # node feature contains 10 origin nf and 16 new_nf (excepted to get delay(RF/EL) and beta etc.)
        x = torch.cat([last_nf, edges.src['nf'], edges.dst['nf']], dim=1)

        # out dim = 8, excepted to get new AT and slew (single path, impossible to use max operation, fanout -> fanin)
        x = self.MLP_netprop(x)
        return {'efn': x}

    def edge_msg_cell(self, edges, groundtruth=False):
        # edges: (one cell) fanin -- cell_out -- fanout

        # generate lut axis query
        if groundtruth:
            last_nf = edges.src['n_atslew']  # input transition(slew) and AT
        else:
            last_nf = edges.src['new_nf']

        # last_nf contains input transition and fanout node features contains output capacitance
        # thus could make lut axis query
        q = torch.cat([last_nf, edges.src['nf'], edges.dst['nf']], dim=1)
        q = self.MLP_lut_query(q)
        q = q.reshape(-1, 2)  # ?
        
        # answer lut axis query
        axis_len = self.in_cell_num_luts * (1 + 2 * self.in_cell_lut_sz)
        axis = edges.data['ef'][:, :axis_len]
        axis = axis.reshape(-1, 1 + 2 * self.in_cell_lut_sz)
        axis = axis.repeat(1, self.lut_dup).reshape(-1, 1 + 2 * self.in_cell_lut_sz)
        a = self.MLP_lut_attention(torch.cat([q, axis], dim=1))
        
        # transform answer to answer mask matrix
        a = a.reshape(-1, 2, self.in_cell_lut_sz)
        ax, ay = torch.split(a, [1, 1], dim=1)
        a = torch.matmul(ax.reshape(-1, self.in_cell_lut_sz, 1), ay.reshape(-1, 1, self.in_cell_lut_sz))  # batch tensor product

        # look up answer matrix in lut
        tables_len = self.in_cell_num_luts * self.in_cell_lut_sz ** 2
        tables = edges.data['ef'][:, axis_len:axis_len + tables_len]
        r = torch.matmul(tables.reshape(-1, 1, 1, self.in_cell_lut_sz ** 2), a.reshape(-1, 4, self.in_cell_lut_sz ** 2, 1))   # batch dot product

        # construct final msg
        r = r.reshape(len(edges), self.in_cell_num_luts * self.lut_dup)
        x = torch.cat([last_nf, edges.src['nf'], edges.dst['nf'], r], dim=1)
        x = self.MLP_cellarc_msg(x)
        k, f1, f2, cef = torch.split(x, [1, self.h1, self.h2, self.out_cef], dim=1)
        k = torch.sigmoid(k)
        return {'efc1': f1 * k, 'efc2': f2 * k, 'efce': cef}

    def node_reduce_o(self, nodes):
        x = torch.cat([nodes.data['nf'], nodes.data['nfc1'], nodes.data['nfc2']], dim=1)
        x = self.MLP_cellreduce(x)
        return {'new_nf': x}

    def node_skip_level_o(self, nodes):
        return {'new_nf': nodes.data['n_atslew']}
        
    def forward(self, g, ts, nf, groundtruth=False):
        """
        @param: g:           graph
        @param: ts:          tensors
        @param: nf:          node features after MLP, shape: nodes, nf+new_nf (10+16)
        @param: groundtruth: bool using true data for training
        """
        assert len(ts['topo']) % 2 == 0, 'The number of logic levels must be even (net, cell, net)'
        
        with g.local_scope():
            # init level 0 with ground truth features
            g.ndata['nf'] = nf  # node features replaced by [node_features, new_nf(from embedding)]
            g.ndata['new_nf'] = torch.zeros(g.num_nodes(), self.out_nf, device='cuda', dtype=nf.dtype)

            # input transition of pi_nodes is known here,
            # meaning input transition is given in SDC?
            g.apply_nodes(self.node_skip_level_o, ts['pi_nodes'])

            def prop_net(nodes, groundtruth):
                # nodes: target node (usually need fanin node)
                # from target nodes, for their in-edge, generate mess on specified edge(by etype)
                # and pull them to target nodes, then aggregate them

                # function:
                g.pull(nodes, functools.partial(self.edge_msg_net, groundtruth=groundtruth), fn.sum('efn', 'new_nf'), etype='net_out')

            def prop_cell(nodes, groundtruth):
                # nodes: fanout (except I pin nodes)
                es = g.in_edges(nodes, etype='cell_out')  # returning in-edge of 'cell_out' etype
                # fanout nodes have two type of in-edge:
                #   fanin node -- cell_out -- fanout node (in one cell)
                #   fanin node -- net_in --fanout node (not in one cell)

                g.apply_edges(functools.partial(self.edge_msg_cell, groundtruth=groundtruth), es, etype='cell_out')
                # maybe use g.pull(etype='cell_out')

                g.send_and_recv(es, fn.copy_e('efc1', 'efc1'), fn.sum('efc1', 'nfc1'), etype='cell_out')
                g.send_and_recv(es, fn.copy_e('efc2', 'efc2'), fn.max('efc2', 'nfc2'), etype='cell_out')
                g.apply_nodes(self.node_reduce_o, nodes)
            
            if groundtruth:
                # don't need to propagate.
                # author says groundtruth is used for training (*****)
                # source: https://github.com/TimingPredict/TimingPredict/issues/11
                # prop_net(ts['input_nodes'], groundtruth)  # input nodes stand for fanin nodes
                # prop_cell(ts['output_nodes_nonpi'], groundtruth)  # for output nodes except I pin node
                for i in range(1, len(ts['topo'])):
                    # i == 0 means I pin node
                    if i % 2 == 1:
                        # i is an odd -> topo[i] is fanin nodes
                        prop_net(ts['topo'][i], groundtruth)
                    else:
                        # i is an even -> topo[i] is fanout nodes
                        prop_cell(ts['topo'][i], groundtruth)

            else:
                # propagate
                for i in range(1, len(ts['topo'])):
                    # i == 0 means I pin node
                    if i % 2 == 1:
                        # i is an odd -> topo[i] is fanin nodes
                        prop_net(ts['topo'][i], groundtruth)
                    else:
                        # i is an even -> topo[i] is fanout nodes
                        prop_cell(ts['topo'][i], groundtruth)
                
            return g.ndata['new_nf'], g.edges['cell_out'].data['efce']


class TimingGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nc1 = NetConv(10, 2, 32)
        self.nc2 = NetConv(32, 2, 32)
        self.nc3 = NetConv(32, 2, 16)  # 16 = 4x delay + 12x arbitrary (might include cap, beta)
        self.prop = SignalProp(10 + 16, 8, 7, 8, 4)

    def forward(self, g, ts, groundtruth=False):
        # why here g has delay info of nodes already
        nf0 = g.ndata['nf']  # node features  shape: nodes, 10
        x = self.nc1(g, ts, nf0)
        x = self.nc2(g, ts, x)
        x = self.nc3(g, ts, x)  # x.shape: nodes, nc3.out_nf(16)
        net_delays = x[:, :4]  # expect the front four element contains info relative to net_delays
        nf1 = torch.cat([nf0, x], dim=1)
        nf2, cell_delays = self.prop(g, ts, nf1, groundtruth=groundtruth)
        return net_delays, cell_delays, nf2

# {AllConv, DeepGCNII}: Simple and Deep Graph Convolutional Networks, arxiv 2007.02133 (GCNII)


class AllConv(torch.nn.Module):
    def __init__(self, in_nf, out_nf, in_ef=12, h1=10, h2=10):
        super().__init__()
        self.h1 = h1
        self.h2 = h2
        self.MLP_msg = MLP(in_nf * 2 + in_ef, 32, 32, 32, 1 + h1 + h2)
        self.MLP_reduce = MLP(in_nf + h1 + h2, 32, 32, 32, out_nf)

    def edge_udf(self, edges):
        x = self.MLP_msg(torch.cat([edges.src['nf'], edges.dst['nf'], edges.data['ef']], dim=1))
        k, f1, f2 = torch.split(x, [1, self.h1, self.h2], dim=1)
        k = torch.sigmoid(k)
        return {'ef1': f1 * k, 'ef2': f2 * k}

    def forward(self, g, nf):   # assume edata is in ef
        with g.local_scope():
            g.ndata['nf'] = nf
            g.apply_edges(self.edge_udf)
            g.update_all(fn.copy_e('ef1', 'ef1'), fn.sum('ef1', 'nf1'))
            g.update_all(fn.copy_e('ef2', 'ef2'), fn.max('ef2', 'nf2'))
            x = torch.cat([g.ndata['nf'], g.ndata['nf1'], g.ndata['nf2']], dim=1)
            x = self.MLP_reduce(x)
            return x


class DeepGCNII(torch.nn.Module):
    def __init__(self, n_layers=60, out_nf=8):
        super().__init__()
        self.n_layers = n_layers
        self.out_nf = out_nf
        self.layer0 = AllConv(10, 16)
        self.layers = [AllConv(26, 16) for i in range(n_layers - 2)]
        self.layern = AllConv(16, out_nf)
        self.layers_store = torch.nn.Sequential(*self.layers)

    def forward(self, g):
        x = self.layer0(g, g.ndata['nf'])
        for layer in self.layers:
            x = layer(g, torch.cat([x, g.ndata['nf']], dim=1)) + x   # both two tricks are mimicked here.
        x = self.layern(g, x)
        return x
