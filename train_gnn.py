import torch
import numpy as np
import dgl
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


parser = argparse.ArgumentParser(description='TimingPredict-GNN')
parser.add_argument(
    '--test', type=bool,default=False,
    help='If specified, executing original code, using to back up that paper')
parser.add_argument(
    '--checkpoint', type=str,default="test",
    help='If specified, the log and model would be saved to/loaded from that checkpoint directory')
parser.set_defaults(netdelay=True, celldelay=True, groundtruth=True)
parser.add_argument(
    '--no_netdelay', dest='netdelay', action='store_false',
    help='Disable the net delay training supervision (default enabled)')
parser.add_argument(
    '--no_celldelay', dest='celldelay', action='store_false',
    help='Disable the cell delay training supervision (default enabled)')
parser.add_argument(
    '--no_groundtruth', dest='groundtruth', action='store_false',
    help='Disable ground-truth breakdown in training (default enabled)')

# using batch size > 1 needs to wrap up more than one graphs and this will change node id
# while id in tensors will not change with it automatically,
# so better to use batch size == 1 (if it is not too slow, need to confirm)
# using several workers need multithread, while using multithread to read one graph always cause error
# so better to use num worker == 0, means using simple thread

parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--batch-size', type=int, default=1, help='batch size')
parser.add_argument('--num-workers',type=int, default=0, help='number of workers')
parser.add_argument('--pin-mem', type=bool, default=False, help='pin memory')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--epochs', type=int, default=1000,help='epoch')
parser.add_argument('--output-dir', type=str, default='res',help='output directory')
parser.add_argument('--opt',type=str, default='default',
                    help='using CircuitNet or not, "default" for paper test, "CircuitNet" for using CircuitNet')
parser.add_argument('--resume',type=str,
                    default='./checkpoints/08_atcd_specul/15799.pth',
                    help='resume checkpoint/weight directory')


# def test(model):    # at
#     model.eval()
#     with torch.no_grad():
#         def test_dict(data):
#             for k, (g, ts) in data.items():
#                 torch.cuda.synchronize()
#                 time_s = time.time()
#                 pred = model(g, ts, groundtruth=False)[2][:, :4]
#                 torch.cuda.synchronize()
#                 time_t = time.time()
#                 truth = g.ndata['n_atslew'][:, :4]
#                 # notice: there is a typo in the parameter order of r2 calculator.
#                 # please see https://github.com/TimingPredict/TimingPredict/issues/7.
#                 # for exact reproducibility of experiments in paper, we will not directly fix the typo here.
#                 # the experimental conclusions are not affected.
#                 # r2 = r2_score(pred.cpu().numpy().reshape(-1),
#                 #               truth.cpu().numpy().reshape(-1))
#                 r2 = r2_score(truth.cpu().numpy().reshape(-1),
#                               pred.cpu().numpy().reshape(-1))
#                 print('{:15} r2 {:1.5f}, time {:2.5f}'.format(k, r2, time_t - time_s))
#
#                 # print('{}'.format(time_t - time_s + ts['topo_time']))
#
#         print('======= Training dataset ======')
#         test_dict(data_train)
#         print('======= Test dataset ======')
#         test_dict(data_test)
#
#
# def test_netdelay(model):    # net delay
#     model.eval()
#     with torch.no_grad():
#         def test_dict(data):
#             for k, (g, ts) in data.items():
#                 pred = model(g, ts, groundtruth=False)[0]
#                 truth = g.ndata['n_net_delays_log']
#                 # notice: there is a typo in the parameter order of r2 calculator.
#                 # please see https://github.com/TimingPredict/TimingPredict/issues/7.
#                 # for exact reproducibility of experiments in paper, we will not directly fix the typo here.
#                 # the experimental conclusions are not affected.
#                 r2 = r2_score(pred.cpu().numpy().reshape(-1),
#                               truth.cpu().numpy().reshape(-1))
#                 print('{:15} {}'.format(k, r2))
#
#         print('======= Training dataset ======')
#         test_dict(data_train)
#         print('======= Test dataset ======')
#         test_dict(data_test)


def train(model, dataloader, optimizer, epoch, args):
    model.train()
    train_loss_tot_net_delays, train_loss_tot_cell_delays, train_loss_tot_ats = 0, 0, 0
    train_loss_tot_cell_delays_prop, train_loss_tot_ats_prop = 0, 0
    optimizer.zero_grad()

    for (g,ts), label in dataloader:
        logging.info(f'training: (data):{label[0]} (size):num_nodes({g.num_nodes()}) num_edges({g.num_edges()})')
        pred_net_delays, pred_cell_delays, pred_atslew = model(g, ts, groundtruth=args.groundtruth)
        loss_net_delays, loss_cell_delays = 0, 0

        if args.netdelay:
            loss_net_delays = F.mse_loss(pred_net_delays, g.ndata['n_net_delays_log'])
            train_loss_tot_net_delays += loss_net_delays.item()

        if args.celldelay:
            loss_cell_delays = F.mse_loss(pred_cell_delays, g.edges['cell_out'].data['e_cell_delays'])
            train_loss_tot_cell_delays += loss_cell_delays.item()
        else:
            # Workaround for a dgl bug...
            # It seems that if some forward propagation channel is not used in backward graph, the GPU memory would BOOM.
            # so we just create a fake gradient channel for this cell delay fork and make sure it does not contribute to gradient by *0.
            loss_cell_delays = torch.sum(pred_cell_delays) * 0.0
        # TODO: compound loss with both groundTruth and propagate, with adaptive coefficient
        loss_ats = F.mse_loss(pred_atslew, g.ndata['n_atslew'])
        train_loss_tot_ats += loss_ats.item()

        (loss_net_delays + loss_cell_delays + loss_ats).backward()

    optimizer.step()
    if epoch % 20 == 0:
        logging.info(f'Epoch {epoch}, training losses: '
                     f'net delay {train_loss_tot_net_delays / train_graph_num:.6f}, '
                     f'cell delay {train_loss_tot_cell_delays / train_graph_num:.6f},'
                     f' at {train_loss_tot_ats / train_graph_num:.6f}')



def validate(model, dataloader, optimizer, epoch, args):
    with torch.no_grad():
        model.eval()
        val_loss_tot_net_delays, val_loss_tot_cell_delays, val_loss_tot_ats = 0, 0, 0
        val_loss_tot_cell_delays_prop, val_loss_tot_ats_prop = 0, 0

        sample_r2 = list()

        for (g, ts), label in dataloader:
            pred_net_delays, pred_cell_delays, pred_atslew = model(g, ts, groundtruth=True)
            _, pred_cell_delays_prop, pred_atslew_prop = model(g, ts, groundtruth=False)

            true_at = g.ndata['n_atslew'][:, :4]  # all nodes, only AT
            pred_at = pred_atslew[:, :4]
            r2 = r2_score(true_at.cpu().numpy().reshape(-1), pred_at.cpu().numpy().reshape(-1))
            sample_r2.append((r2,g.num_nodes()))

            if args.netdelay:
                val_loss_tot_net_delays += F.mse_loss(pred_net_delays, g.ndata['n_net_delays_log']).item()
            if args.celldelay:
                val_loss_tot_cell_delays += F.mse_loss(pred_cell_delays, g.edges['cell_out'].data['e_cell_delays']).item()
            val_loss_tot_ats += F.mse_loss(pred_atslew, g.ndata['n_atslew']).item()
            val_loss_tot_ats_prop += F.mse_loss(pred_atslew_prop, g.ndata['n_atslew']).item()
        logging.info(f'Epoch {epoch}, validate losses: '
                     f'net delay {val_loss_tot_net_delays / validate_graph_num:.6f}, '
                     f'cell delay {val_loss_tot_cell_delays / validate_graph_num:.6f}, '
                     f'at {val_loss_tot_ats / validate_graph_num:.6f}, '
                     f'at prop {val_loss_tot_ats_prop / validate_graph_num:.6f}')
        mean_r2 = 0
        nodes_tot = 0
        for element in sample_r2:
            r2 = element[0]
            num_nodes = element[1]
            mean_r2 += r2 * num_nodes
            nodes_tot += num_nodes
        mean_r2 = mean_r2 / float(nodes_tot)
        return mean_r2

        # print('Epoch {}, net delay {:.6f}/{:.6f}, cell delay {:.6f}/{:.6f}, at {:.6f}/({:.6f}, {:.6f})'.format(
        #     epoch,
        #     train_loss_tot_net_delays / batch_size,
        #     test_loss_tot_net_delays / len(data_test),
        #     train_loss_tot_cell_delays / batch_size,
        #     test_loss_tot_cell_delays / len(data_test),
        #     train_loss_tot_ats / batch_size,
        #     # train_loss_tot_ats_prop / batch_size,
        #     test_loss_tot_ats / len(data_test),
        #     test_loss_tot_ats_prop / len(data_test)))

    # if epoch == 0 or epoch % 200 == 199 or (epoch > 6000 and test_loss_tot_ats_prop / len(data_test) < 6):
    #     if args.checkpoint:
    #         save_path = './checkpoints/{}/{}.pth'.format(args.checkpoint, epoch)
    #         torch.save(model.state_dict(), save_path)
    #         print('saved model to', save_path)
    #     try:
    #         test(model)
    #     except ValueError as e:
    #         print(e)
    #         print('Error testing, but ignored')


def test(model, dataloader, optimizer, args):
    with torch.no_grad():
        model.eval()

        sample_r2 = list()

        for (g, ts), label in dataloader:
            pred_net_delays, pred_cell_delays, pred_atslew = model(g, ts, groundtruth=True)
            pred_net_delays_prop, pred_cell_delays_prop, pred_atslew_prop = model(g, ts, groundtruth=False)
            r2 = [0,0,0,0,0,0]
            true_at = g.ndata['n_atslew'][:, :4]  # all nodes, only AT
            true_net_delay = g.ndata['n_net_delays_log']
            true_cell_delay = g.edges['cell_out'].data['e_cell_delays']
            pred_at_prop = pred_atslew_prop[:, :4]
            pred_at = pred_atslew[:, :4]
            r2[0] = r2_score(true_at.cpu().numpy().reshape(-1), pred_at.cpu().numpy().reshape(-1))
            r2[1] = r2_score(true_at.cpu().numpy().reshape(-1), pred_at_prop.cpu().numpy().reshape(-1))
            sample_r2.append((r2,g.num_nodes()))

            r2[2] = r2_score(true_net_delay.cpu().numpy().reshape(-1), pred_net_delays.cpu().numpy().reshape(-1))
            r2[3] = r2_score(true_net_delay.cpu().numpy().reshape(-1), pred_net_delays_prop.cpu().numpy().reshape(-1))
            r2[4] = r2_score(true_cell_delay.cpu().numpy().reshape(-1), pred_cell_delays.cpu().numpy().reshape(-1))
            r2[5] = r2_score(true_cell_delay.cpu().numpy().reshape(-1), pred_cell_delays_prop.cpu().numpy().reshape(-1))
            print(f"{label}:  nodes:{g.num_nodes()}\n"
                  f"\tAT R2:{r2[0]:.8f}                AT_prop R2:{r2[1]:.8f}\n"
                  f"\tnet_delay R2:{r2[2]:.8f}         net_delay_prop R2:{r2[3]:.8f}\n"
                  f"\tcell_delay R2:{r2[4]:.8f}        cell_delay_prop R2:{r2[5]:.8f}\n"
                  f"\n"
                  )

        mean_r2 = 0
        nodes_tot = 0
        for element in sample_r2:
            r2 = element[0][1]
            num_nodes = element[1]
            mean_r2 += r2 * num_nodes
            nodes_tot += num_nodes
        mean_r2 = mean_r2 / float(nodes_tot)
        return mean_r2

if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    logging.basicConfig(
        filename=os.path.join(args.output_dir,'TimingGCN.log'),
        format='%(asctime)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(args)

    device = torch.device(args.device)

    model = TimingGCN()
    model.to(device)

    if args.test:
        checkpoint = torch.load('./checkpoints/08_atcd_specul/15799.pth')
        model.load_state_dict(checkpoint, strict=True)
        train_graph_num = 12
        validate_graph_num = 5

        dataloader_train, dataloader_validate = get_dataloder(dgl_graphs_path, labels,
                                                              train_graph_num, validate_graph_num, args)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # r2 = test(model, dataloader_train, optimizer, args)
        # r2 = test(model, dataloader_validate, optimizer, args)
        for epoch in range(1):
            train(model, dataloader_train, optimizer, epoch, args)
        r2 = test(model, dataloader_validate, optimizer, args)
        print(f"meen R2:{r2}")

    else:
        print(f"training model TimingGCN on device {args.device}")
        print('saving logs and models to ./checkpoints/{}'.format(args.checkpoint))

        if not os.path.exists(f'./checkpoints/{args.checkpoint}'):
            os.makedirs('./checkpoints/{}'.format(args.checkpoint))  # exist not ok

        train_graph_num = 12
        validate_graph_num = 5

        dataloader_train, dataloader_validate = get_dataloder(dgl_graphs_path, labels,
                                                              train_graph_num, validate_graph_num, args)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        if args.resume is not None:
            if not os.path.exists(args.resume):
                logging.error('resume directory not exist')
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint, strict=True)

        print(f"Start training for {args.epochs} epochs")
        # TODO: use r2 score to save weight for best epoch
        best_epoch = 0
        best_r2 = 0
        save_path = './checkpoints/{}/best_r2.pth'.format(args.checkpoint)
        print('saved model to', save_path)
        for epoch in range(args.epochs):
            train(model, dataloader_train, optimizer, epoch, args)
            if epoch % 20 == 0:
                r2 = validate(model,dataloader_validate,optimizer,epoch, args)
                if r2 > best_r2:
                    best_r2 = r2
                    best_epoch = epoch
                    torch.save(model.state_dict(), save_path)
                logging.info(f'Epoch {epoch}, validate R2: {r2:.6f}')
                logging.info(f'Best R2 at Epoch {best_epoch}, best R2: {best_r2:.6f}')






