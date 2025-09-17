import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pdb
import argparse
import torch
from model import TimingGCN
from Mydataset import get_dataloder
from data_graph import labels, dgl_graphs_path

matplotlib.use('TkAgg')
parser = argparse.ArgumentParser(description='TimingPredict-GNN')
parser.add_argument('--batch-size', type=int, default=1, help='batch size')
parser.add_argument('--num-workers',type=int, default=0, help='number of workers')
parser.add_argument('--pin-mem', type=bool, default=False, help='pin memory')
parser.add_argument('--opt',type=str, default='default')
parser.add_argument('--checkpoint',type=str,
                    default=r'E:\ssta_gnn\TimingPredict-master\TimingPredict-master\linux\model_ours2\checkpoint_best_r2_prop.pth.tar')
parser.add_argument('--outputdir',type=str,default='pic/groundtruth')

def plot(pred, truth, label, type, save_path):
    if type == 'at':
        title = f"arrival time prediction {label}"
    elif type == 'cell_delay':
        title = f"cell delay prediction {label}"
    elif type == 'net_delay':
        title = f"net delay prediction {label}"
    else:
        raise ValueError(f"Unsupported type of {type}")
    plt.title(title)
    plt.plot(truth,truth,label='truth',c='black')
    plt.scatter(truth,pred,label='prediction',c='red',s=1,alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(save_path, f"{title}.png"), dpi=300)
    plt.close()


if __name__ == '__main__':
    args = parser.parse_args()
    model = TimingGCN()
    checkpoint = torch.load(args.checkpoint,map_location='cpu')
    if '15799' in args.checkpoint:
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint['model'])
    model.cuda()
    model.eval()
    train_graph_num = 12
    validate_graph_num = 5
    args.train_graph_num = train_graph_num
    args.validate_graph_num = validate_graph_num
    dataloader_train, dataloader_validate = get_dataloder(dgl_graphs_path, labels,
                                                          train_graph_num, validate_graph_num, args)
    save_path = os.path.join(args.outputdir, 'ours')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with torch.no_grad():
        for (g,ts), label in dataloader_train:
            if isinstance(label, tuple) and len(label) == 1:
                label = label[0]
            assert isinstance(label, str), [f'TypeError, received label with type {type(label)}, {label}']
            print(label, "  ", g.num_nodes())
            pred_net_delays, pred_cell_delays, pred_atslew, _, _ = model(g, ts, groundtruth=1.0)
            # pred_net_delays_prop, pred_cell_delays_prop, pred_atslew_prop, _, _ = model(g, ts, groundtruth=0)
            truth = g.ndata['n_atslew'][:,:4].reshape(-1).cpu().numpy().tolist()
            pred = pred_atslew[:,:4].reshape(-1).cpu().numpy().tolist()
            plot(pred, truth,label,'at',save_path)
            truth = g.edges['cell_out'].data['e_cell_delays'].reshape(-1).cpu().numpy().tolist()
            pred = pred_cell_delays.reshape(-1).cpu().numpy().tolist()
            plot(pred, truth, label, 'cell_delay', save_path)
        for (g,ts), label in dataloader_validate:
            if isinstance(label, tuple) and len(label) == 1:
                label = label[0]
            assert isinstance(label, str), [f'TypeError, received label with type {type(label)}, {label}']
            print(label, "  ", g.num_nodes())
            pred_net_delays, pred_cell_delays, pred_atslew, _, _ = model(g, ts, groundtruth=1.0)
            # pred_net_delays_prop, pred_cell_delays_prop, pred_atslew_prop, _, _ = model(g, ts, groundtruth=0)
            truth = g.ndata['n_atslew'][:,:4].reshape(-1).cpu().numpy().tolist()
            pred = pred_atslew[:,:4].reshape(-1).cpu().numpy().tolist()
            plot(pred, truth,label,'at',save_path)
            truth = g.edges['cell_out'].data['e_cell_delays'].reshape(-1).cpu().numpy().tolist()
            pred = pred_cell_delays.reshape(-1).cpu().numpy().tolist()
            plot(pred, truth, label, 'cell_delay', save_path)






# for b in ['usbf_device']:
#     data = np.load('checkpoints/18_slacksdump/{}.npz'.format(b))
#     se, sl, se_truth, sl_truth = [data['arr_{}'.format(i)].flatten() for i in range(4)]
#
#     def plot_slack(ax, se_truth, se, title):
#         # pdb.set_trace()
#         ax.set_title(title)
#         ax.scatter(se_truth, se, s=10)
#         ax.set_xlabel('Truth')
#         ax.set_ylabel('Predicted')
#         # line = np.polyfit(se_truth, se, 1)
#         # y1 = line[0] * se_truth + line[1]
#         # ax.plot(se_truth, y1, 'r-')
#         ax.plot(se_truth, se_truth, 'r-')
#
#     fig, ax = plt.subplots(1, 2)
#     # fig.tight_layout()
#     plt.subplots_adjust(left=0.125, bottom=0.2, right=0.9, top=0.9, wspace=0.4, hspace=0.2)
#     plot_slack(ax[0], se_truth, se, 'Hold slacks')
#     ax[0].xaxis.set_ticks([-5, -2.5, 0, 2.5, 5])
#     ax[0].yaxis.set_ticks([-5, -2.5, 0, 2.5, 5])
#     plot_slack(ax[1], sl_truth, sl, 'Setup slacks')
#     ax[1].xaxis.set_ticks([-10, -5, 0, 5, 10])
#     ax[1].yaxis.set_ticks([-10, -5, 0, 5, 10])
#     fig.canvas.manager.set_window_title(b)
#     plt.show()
