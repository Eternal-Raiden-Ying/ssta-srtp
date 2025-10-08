import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import logging
import pdb
import time
import argparse
import os
from sklearn.metrics import r2_score

from data_graph import labels, dgl_graphs_path
from model import TimingGCN
from Mydataset import DGLGraphDataset, get_dataloder
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from utils import *

plt.rcParams['font.family'] = ['SimSun', 'Times New Roman']  # 宋体 + Times New Roman
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.unicode_minus'] = False


parser = argparse.ArgumentParser(description='TimingPredict-GNN')
parser.add_argument(
    '--test', type=bool,default=True,
    help='If specified, executing original code, using to back up that paper')
parser.add_argument(
    '--checkpoint', type=str,default="model_v11_gradient_clip_norm_data",
    help='If specified, the log and model would be saved to/loaded from that checkpoint directory,'
         'currently loaded from fixed directory')
parser.set_defaults(netdelay=True, celldelay=True, groundtruth=1.0)
parser.set_defaults(process_data=True)
parser.add_argument(
    '--no_netdelay', dest='netdelay', action='store_false',
    help='Disable the net delay training supervision (default enabled)')
parser.add_argument(
    '--no_celldelay', dest='celldelay', action='store_false',
    help='Disable the cell delay training supervision (default enabled)')

parser.add_argument('--batch-size', type=int, default=1, help='batch size')
parser.add_argument('--norm', type=bool, default=False, help='use normalize (default False)')
parser.add_argument('--num-workers',type=int, default=0, help='number of workers')
parser.add_argument('--pin-mem', type=bool, default=False, help='pin memory')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--epochs', type=int, default=1000,help='epoch')
parser.add_argument('--start-epoch',type=int,default=0,help='start epoch')
parser.add_argument('--output-dir', type=str, default='res',help='output directory')
parser.add_argument('--enable-process-data', dest='process_data', action='store_true',
    help='Enable saving process data (default Disabled)')
parser.add_argument('--frequency',type=int,default=1)
parser.add_argument('--opt',type=str, default='default',
                    help='using CircuitNet or not, "default" for paper test, "CircuitNet" for using CircuitNet')
parser.add_argument('--resume',type=str,
                    # default=r"E:\ssta_gnn\TimingPredict-master\TimingPredict-master\res\model_v5__dynamic_groundtruth\checkpoint_best_r2.pth.tar",
                    # default=r"E:\ssta_gnn\TimingPredict-master\TimingPredict-master\res\checkpoint_best_r2_model_v2.pth.tar",
                    # default='./checkpoints/08_atcd_specul/15799.pth',
                    default=None,
                    help='resume checkpoint/weight directory')

# Learning rate schedule parameters
parser.add_argument('--sched', default='plateau', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "plateau"')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-6)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.05, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=1e-3, metavar='LR',
                    help='warmup learning rate (default: 1e-6)')
parser.add_argument('--min-lr', type=float, default=1e-7, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--max-lr', type=float, default=1e-2, metavar='LR')
parser.add_argument('--threshold',type=float,default=0.05)
parser.add_argument('--increase-factor',type=float,default=1.2)

parser.add_argument('--decay-epochs', type=float, default=20, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.5, metavar='RATE',
                    help='LR decay rate (default: 0.5)')
parser.add_argument('--plateau_mode',type=str,default='min',help='plateau-mode ("min" or "max")')
parser.add_argument('--beta1',type=float,default=0.9)
parser.add_argument('--beta2',type=float,default=0.999)


def plot_slack(ax, truth, pred, title, pth=None):
    ax.set_title(title)
    ax.scatter(truth, pred, s=2, color='red')
    ax.plot(truth,truth, color='black')
    ax.set_xlabel('Truth')
    ax.set_ylabel('Predicted')



def test(model, dataloader, optimizer, args):
    with torch.no_grad():
        model.eval()
        args.norm = False
        hold = np.array([0, 1])
        setup = np.array([2, 3])

        for (g, ts), label in dataloader:
            pred_net_delays, pred_cell_delays, pred_atslew, _, _ = model(g, ts, groundtruth=True)
            true_at = g.ndata['n_atslew'][:, :4].detach().clone().cpu().numpy()
            pred_at = pred_atslew[:,:4].detach().clone().cpu().numpy()
            rt = g.ndata['n_rats'].detach().clone().cpu().numpy()
            true_slack_setup = rt[:,setup] - true_at[:,setup]
            pred_slack_setup = rt[:,setup] - pred_at[:,setup]
            true_slack_hold = true_at[:,hold] - rt[:,hold]
            pred_slack_hold = pred_at[:,hold] - rt[:,hold]

            fig, ax = plt.subplots(1, 2,figsize=(8,5))
            # fig.tight_layout()
            plt.subplots_adjust(left=0.125, bottom=0.2, right=0.9, top=0.9, wspace=0.4, hspace=0.2)
            plot_slack(ax[0], true_slack_hold, pred_slack_hold, 'Hold slacks')
            ax[0].xaxis.set_ticks([-2.5, 0, 2.5, 5])
            ax[0].yaxis.set_ticks([-2.5, 0, 2.5, 5])
            plot_slack(ax[1], true_slack_setup, pred_slack_setup, 'Setup slacks')
            ax[1].xaxis.set_ticks([-5, 0, 5, 10,15,20])
            ax[1].yaxis.set_ticks([-5, 0, 5, 10,15,20])
            fig.suptitle(label[0])
            plt.show()



if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.exists(os.path.join(args.output_dir,args.checkpoint)):
        os.makedirs(os.path.join(args.output_dir,args.checkpoint))
    logging.basicConfig(
        filename=os.path.join(args.output_dir,f'{args.checkpoint}/TimingGCN.log'),
        format='%(asctime)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(args)

    device = torch.device(args.device)

    model = TimingGCN()
    model.to(device)

    checkpoint = torch.load('./checkpoints/08_atcd_specul/15799.pth')
    model.load_state_dict(checkpoint, strict=True)
    train_graph_num = 12
    validate_graph_num = 5

    dataloader_train, dataloader_validate = get_dataloder(dgl_graphs_path, labels,
                                                          train_graph_num, validate_graph_num, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    test(model, dataloader_train, optimizer, args)
    test(model, dataloader_validate, optimizer, args)







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
