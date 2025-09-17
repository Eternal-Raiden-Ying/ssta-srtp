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
parser.add_argument('--model',type=str, default='ours')

def add_noise_to_atslew(graph, noise_range, noise_pct=0.01):
    """
        noise_range(float):噪声覆盖范围，默认0.5
        noise_pct (float): 噪声占比，默认 0.01。
    """
    # 确保输入 tensor 是 float 类型
    assert 'n_atslew' in graph.ndata.keys()

    raw_atslew = graph.ndata['n_atslew'].detach()
    tensor = graph.ndata['n_atslew']
    with torch.no_grad():
        num_elements = tensor.numel()
        num_noisy_elements = int(num_elements * noise_range)

        noisy_indices = torch.randperm(num_elements)[:num_noisy_elements].cuda()

        noise_factor = torch.randn(tensor.shape).cuda()

        mask = torch.ones(tensor.shape, dtype=torch.bool,device='cuda:0')  # 默认是 float，用 bool 更节省内存
        s = mask.shape
        mask.reshape(-1).index_fill_(0, noisy_indices, False)  # 在 dim=0 的 indices 处填充 True（即 1）
        noise_factor[mask.reshape(s)] = 0

        tensor = tensor + tensor * noise_factor * noise_pct

    return tensor, raw_atslew


def analysis_input_stability(model, graph, ts, label, args):
    print("validate input stability")
    save_path = os.path.join(args.output_dir,f"{label}/invalid")
    normal_save_path = os.path.join(args.output_dir,f"{label}/normal")
    if not os.path.exists(os.path.join(save_path,"atslew")):
        os.makedirs(os.path.join(save_path,"atslew"))
        os.makedirs(os.path.join(save_path,"cell_delay"))
    if not os.path.exists(normal_save_path):
        os.makedirs(os.path.join(normal_save_path,"atslew"))
        os.makedirs(os.path.join(normal_save_path,"cell_delay"))
    with torch.no_grad():
        pred_net_delays, pred_cell_delays, pred_atslew, node_topo, cell_topo = model(graph, tensors, False)
        truth = graph.ndata['n_atslew']
        r2 = r2_score(truth.cpu().numpy().reshape(-1), pred_atslew.cpu().numpy().reshape(-1))
        print(f"{label}: no noise r2: {r2:.15f}")
    with torch.no_grad():
        for i in range(100,120):
            noise_pct = 0.01 * ((i//20)+1)
            noise_pct = 0.05
            graph.ndata['n_atslew'], raw_atslew = add_noise_to_atslew(graph, 1, noise_pct)
            pred_net_delays, pred_cell_delays, pred_atslew, node_topo, cell_topo = model(graph, tensors, False)
            truth = raw_atslew
            if pred_atslew.isnan().any():
                torch.save(
                    torch.cat([pred_atslew, raw_atslew, node_topo], dim=1),
                    os.path.join(save_path,f'atslew/noise{noise_pct}_{i}.pt'))
                pred_atslew[pred_atslew.isnan()] = 0
                print(torch.max(pred_atslew), torch.min(pred_atslew))
            else:
                torch.save(
                    torch.cat([pred_atslew, raw_atslew, node_topo], dim=1),
                    os.path.join(normal_save_path, f'atslew/noise{noise_pct}_{i}.pt'))
                r2 = r2_score(truth.cpu().numpy().reshape(-1), pred_atslew.cpu().numpy().reshape(-1))
                print(f"{label}: noise_pct:{noise_pct} at_slew r2: {r2:.15f}")

            if pred_cell_delays.isnan().any():
                torch.save(
                    torch.cat([pred_cell_delays, graph.edges['cell_out'].data['e_cell_delays'], cell_topo], dim=1),
                    os.path.join(save_path,f'cell_delay/noise{noise_pct}_{i}.pt'))
                pred_cell_delays[pred_cell_delays.isnan()] = 0
                print(torch.max(pred_cell_delays), torch.min(pred_cell_delays))
            else:
                torch.save(
                    torch.cat([pred_atslew, raw_atslew, node_topo], dim=1),
                    os.path.join(normal_save_path, f'cell_delay/noise{noise_pct}_{i}.pt'))
                r2 = r2_score(graph.edges['cell_out'].data['e_cell_delays'].cpu().numpy().reshape(-1),
                              pred_cell_delays.cpu().numpy().reshape(-1))
                print(f"{label}: noise_pct:{noise_pct} cell_delay r2: {r2:.15f}")

            graph.ndata['n_atslew'] = raw_atslew


# def analysis_input_stability(model, graph, ts, label, args):
#     print("validate input stability")
#     with torch.no_grad():
#         pred_net_delays, pred_cell_delays, pred_atslew, node_topo, cell_topo = model(graph, tensors, False)
#         truth = graph.ndata['n_atslew']
#         r2 = r2_score(truth.cpu().numpy().reshape(-1), pred_atslew.cpu().numpy().reshape(-1))
#         print(f"{label}: no noise r2: {r2:.15f}")
#     with torch.no_grad():
#         for i in range(20):
#             noise_pct = 0.05
#             graph.ndata['n_atslew'], raw_atslew = add_noise_to_atslew(graph, 1, noise_pct)
#             pred_net_delays, pred_cell_delays, pred_atslew, node_topo, cell_topo = model(graph, tensors, False)
#             truth = raw_atslew
#             if pred_atslew.isnan().any():
#                 pred_atslew[pred_atslew.isnan()] = 0
#                 print(torch.max(pred_atslew), torch.min(pred_atslew))
#             else:
#                 r2 = r2_score(truth.cpu().numpy().reshape(-1), pred_atslew.cpu().numpy().reshape(-1))
#                 print(f"{label}: noise_pct:{noise_pct} at_slew r2: {r2:.15f}")
#
#             if pred_cell_delays.isnan().any():
#                 pred_cell_delays[pred_cell_delays.isnan()] = 0
#                 print(torch.max(pred_cell_delays), torch.min(pred_cell_delays))
#             else:
#                 r2 = r2_score(graph.edges['cell_out'].data['e_cell_delays'].cpu().numpy().reshape(-1),
#                               pred_cell_delays.cpu().numpy().reshape(-1))
#                 print(f"{label}: noise_pct:{noise_pct} cell_delay r2: {r2:.15f}")
#
#             graph.ndata['n_atslew'] = raw_atslew

def compute_condition_number(J):
    U, S, V = torch.svd(J)  # 奇异值分解
    cond_number = S.max() / S.min()
    return {"cv":float(cond_number),"singular =_values":S.cpu().numpy().tolist()}


def power_iteration(J, num_iters=100, tol=1e-6):
    """幂迭代法估计最大特征值"""
    b_k = torch.randn(J.shape[-1], device=J.device).reshape(-1,1)
    b_k = b_k / torch.norm(b_k)

    prev_value = 0
    b_k1_norm = None
    for _ in range(num_iters):
        b_k1 = torch.matmul(J, b_k)
        b_k1_norm = torch.norm(b_k1)
        b_k = b_k1 / b_k1_norm

        # 检查收敛性
        if abs(b_k1_norm - prev_value) < tol:
            break
        prev_value = b_k1_norm

    return b_k1_norm  # 返回最大特征值


if __name__ == "__main__":
    print("model numerical analysis")
    args = parser.parse_args()
    checkpoint_path = r"E:\ssta_gnn\TimingPredict-master\TimingPredict-master\checkpoints\08_atcd_specul\15799.pth"
    model = TimingGCN()
    signal_prop = model.prop
    checkpoint = torch.load(checkpoint_path,map_location='cpu')
    model.load_state_dict(checkpoint,strict=True)
    assert torch.cuda.is_available(), 'cuda is not available'
    model.cuda()

    # data_loader, _ = get_dataloder(dgl_graphs_path,labels,15,0, args=args)
    # data_loader, _ = get_dataloder(['data/8_rat/spm.graph.bin',], ['spm',], 1, 0, args=args)
    data_loader, _ = get_dataloder(['data/8_rat/blabla.graph.bin',], ['blabla',], 1, 0, args=args)

    for (graph, tensors), label in data_loader:
        if isinstance(label, tuple) and len(label) == 1:
            label = label[0]
        assert isinstance(label, str), [f'TypeError, received label with type {type(label)}, {label}']
        print(label, "  ", graph.num_nodes())
        # NOTICE: Here we need to analysis why groundtruth differentiate greatly from propagate
        # as a result, we need to give a slight noise to input features
        # to see if there is great changes in output (for trained weight)
        # rather than calculate the gradient of input features

        analysis_input_stability(model, graph, tensors, label, args)

    # for name, parameter in model.named_parameters():
    #     print(name, parameter.shape)

    optimizer = torch.optim.Adam(model.parameters())
    norm_dict = dict()
    Jcobian_cv_dict = dict()
    var_scale_dict = dict()
    radius_dict = dict()
    Hessian_cv_dict = dict()
    for (graph, tensors), label in data_loader:
        label = label[0]
        print(label, "  ", graph.num_nodes())

        # mse, var_scale
        optimizer.zero_grad()
        pred_net_delays, pred_cell_delays, pred_atslew, node_topo, cell_topo = model(graph, tensors, False)
        loss = torch.nn.functional.mse_loss(pred_atslew, graph.ndata['n_atslew'])
        loss.backward(retain_graph=True)
        with torch.no_grad():
            if "mse_loss" not in norm_dict.keys():
                norm_dict['mse_loss'] = dict()
                Jcobian_cv_dict['mse_loss'] = dict()
                radius_dict['mse_loss'] = dict()
            var_scale_dict[label] = dict()
            Jcobian_cv_dict['mse_loss'][label] = dict()
            # radius_dict['mse_loss'][label] = dict()
            for name, parameter in model.named_parameters():
                if 'weight' not in name: continue
                if name in norm_dict['mse_loss'].keys():
                    norm_dict['mse_loss'][name] += float(torch.norm(parameter.grad))
                else:
                    norm_dict['mse_loss'][name] = float(torch.norm(parameter.grad))
                var_scale_dict[label][name] = {'max':float(torch.max(torch.abs(parameter.grad/(parameter + 1e-10)))),
                                               'mean':float(torch.mean(torch.abs(parameter.grad / (parameter + 1e-10)))),
                                               'min':float(torch.min(torch.abs(parameter.grad/(parameter + 1e-10))))}
                Jcobian_cv_dict['mse_loss'][label][name] = compute_condition_number(parameter.grad)
                # radius_dict['mse_loss'][label][name] = power_iteration(parameter.grad)

        # mae
        optimizer.zero_grad()
        pred_net_delays, pred_cell_delays, pred_atslew, node_topo, cell_topo = model(graph, tensors, False)
        loss = mae_loss(pred_atslew, graph.ndata['n_atslew'])
        loss.backward(retain_graph=True)
        with torch.no_grad():
            if "mae_loss" not in norm_dict.keys():
                norm_dict['mae_loss'] = dict()
                Jcobian_cv_dict['mae_loss'] = dict()
                radius_dict['mae_loss'] = dict()
            Jcobian_cv_dict['mae_loss'][label] = dict()
            # radius_dict['mae_loss'][label] = dict()
            for name, parameter in model.named_parameters():
                if 'weight' not in name: continue
                if name in norm_dict['mae_loss'].keys():
                    norm_dict['mae_loss'][name] += float(torch.norm(parameter.grad))
                else:
                    norm_dict['mae_loss'][name] = float(torch.norm(parameter.grad))
                Jcobian_cv_dict['mae_loss'][label][name] = compute_condition_number(parameter.grad)
                # radius_dict['mae_loss'][label][name] = power_iteration(parameter.grad)

        # log_cosh
        optimizer.zero_grad()
        pred_net_delays, pred_cell_delays, pred_atslew, node_topo, cell_topo = model(graph, tensors, False)
        loss = log_cosh_loss(pred_atslew, graph.ndata['n_atslew'])
        loss.backward(retain_graph=True)
        with torch.no_grad():
            if "log_cosh_loss" not in norm_dict.keys():
                norm_dict['log_cosh_loss'] = dict()
                Jcobian_cv_dict['log_cosh_loss'] = dict()
                radius_dict['log_cosh_loss'] = dict()
            Jcobian_cv_dict['log_cosh_loss'][label] = dict()
            # radius_dict['log_cosh_loss'][label] = dict()
            for name, parameter in model.named_parameters():
                if 'weight' not in name: continue
                if name in norm_dict['log_cosh_loss'].keys():
                    norm_dict['log_cosh_loss'][name] += float(torch.norm(parameter.grad))
                else:
                    norm_dict['log_cosh_loss'][name] = float(torch.norm(parameter.grad))
                Jcobian_cv_dict['mae_loss'][label][name] = compute_condition_number(parameter.grad)
                # radius_dict['mae_loss'][label][name] = power_iteration(parameter.grad)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir,args.model,'norm.json'),'w') as file:
        json.dump(norm_dict,file,indent=4)

    with open(os.path.join(args.output_dir,args.model,'jacobian_cv.json'),'w') as file:
        json.dump(Jcobian_cv_dict,file,indent=4)

    with open(os.path.join(args.output_dir,args.model,'var_scale.json'),'w') as file:
        json.dump(var_scale_dict,file,indent=4)


    # print("norm:\n",norm_dict,"\n\n")
    # print("jacobian_cv:\n",Jcobian_cv_dict,"\n\n")
    # print("radius:\n",radius_dict,"\n\n")
    # print("var_scale:\n",var_scale_dict)


