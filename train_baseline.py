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
from utils import *


parser = argparse.ArgumentParser(description='TimingPredict-GNN')
parser.add_argument(
    '--test', type=bool,default=False,
    help='If specified, executing original code, using to back up that paper')
parser.add_argument(
    '--checkpoint', type=str,default="model_v9_log_loss",
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

# using batch size > 1 needs to wrap up more than one graphs and this will change node id
# while id in tensors will not change with it automatically,
# so better to use batch size == 1 (if it is not too slow, need to confirm)
# using several workers need multithread, while using multithread to read one graph always cause error
# so better to use num worker == 0, means using simple thread

parser.add_argument('--batch-size', type=int, default=1, help='batch size')
parser.add_argument('--num-workers',type=int, default=0, help='number of workers')
parser.add_argument('--pin-mem', type=bool, default=False, help='pin memory')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--epochs', type=int, default=30000,help='epoch')
parser.add_argument('--start-epoch',type=int,default=0,help='start epoch')
parser.add_argument('--output-dir', type=str, default='res',help='output directory')
parser.add_argument('--enable-process-data', dest='process_data', action='store_true',
    help='Enable saving process data (default Disabled)')
parser.add_argument('--frequency',type=int,default=50)
parser.add_argument('--opt',type=str, default='default',
                    help='using CircuitNet or not, "default" for paper test, "CircuitNet" for using CircuitNet')
parser.add_argument('--resume',type=str,
                    default=None,
                    help='resume checkpoint/weight directory')

# Learning rate schedule parameters
parser.add_argument('--sched', default='plateau', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "plateau"')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
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
parser.add_argument('--beta1',type=float,default=0.85)
parser.add_argument('--beta2',type=float,default=0.995)


def train(model, dataloader, optimizer, epoch, groundtruth, args):
    model.train()
    train_loss_tot_net_delays, train_loss_tot_cell_delays, train_loss_tot_ats = 0, 0, 0
    optimizer.zero_grad()
    for (g,ts), label in dataloader:
        # logging.info(f'training: (data):{label[0]} (size):num_nodes({g.num_nodes()}) num_edges({g.num_edges()})')
        pred_net_delays, pred_cell_delays, pred_atslew, node_topo, cell_topo = model(g, ts, groundtruth=groundtruth)
        loss_net_delays, loss_cell_delays = 0, 0
        max_layer = node_topo.max()

        if args.netdelay:
            loss_net_delays = F.mse_loss(pred_net_delays, g.ndata['n_net_delays_log'])
            # loss_net_delays.backward(retain_graph=True)
            train_loss_tot_net_delays += loss_net_delays.item()

        if args.celldelay:
            loss_cell_delays = F.mse_loss(pred_cell_delays, g.edges['cell_out'].data['e_cell_delays'])
            # loss_cell_delays.backward(retain_graph=True)
            train_loss_tot_cell_delays += loss_cell_delays.item()

        else:
            # Workaround for a dgl bug...
            # It seems that if some forward propagation channel is not used in backward graph, the GPU memory would BOOM.
            # so we just create a fake gradient channel for this cell delay fork and make sure it does not contribute to gradient by *0.
            loss_cell_delays = torch.sum(pred_cell_delays) * 0.0

        loss_ats = F.mse_loss(pred_atslew, g.ndata['n_atslew'])
        train_loss_tot_ats += loss_ats.item()
        (loss_net_delays + loss_cell_delays + loss_ats).backward()
    optimizer.step()

    if epoch % args.frequency == 0:
        with open(os.path.join(args.output_dir,args.checkpoint,'train_loss.txt'),'a') as file:
            file.write(f'{epoch},'
                       f'{train_loss_tot_net_delays / args.train_graph_num:.15f},'
                       f'{train_loss_tot_cell_delays / args.train_graph_num:.15f},'
                       f'{train_loss_tot_ats / args.train_graph_num:.15f}\n')

    if epoch % args.frequency == 0:
        lr = [group['lr'] for group in optimizer.param_groups]
        logging.info(f'Epoch {epoch}, lr {lr}, training losses: '
                     f'net delay {train_loss_tot_net_delays / args.train_graph_num:.8f}, '
                     f'cell delay {train_loss_tot_cell_delays / args.train_graph_num:.8f},'
                     f' at {train_loss_tot_ats / args.train_graph_num:.8f}')
    return (train_loss_tot_net_delays / args.train_graph_num,
            train_loss_tot_cell_delays / args.train_graph_num,
            train_loss_tot_ats / args.train_graph_num)

def validate(model, dataloader, optimizer, epoch, args):
    with torch.no_grad():
        model.eval()
        val_loss_tot_net_delays, val_loss_tot_cell_delays, val_loss_tot_ats = 0, 0, 0
        val_loss_tot_cell_delays_prop, val_loss_tot_ats_prop = 0, 0

        sample_r2 = list()

        for (g, ts), label in dataloader:
            pred_net_delays, pred_cell_delays, pred_atslew,_,_ = model(g, ts, groundtruth=1.0)
            pred_net_delays_prop, pred_cell_delays_prop, pred_atslew_prop,_,_ = model(g, ts, groundtruth=0)

            true_at = g.ndata['n_atslew'][:, :4]  # all nodes, only AT
            pred_at = pred_atslew[:, :4]
            pred_at_prop = pred_atslew_prop[:, :4]
            r2 = [0,0]
            if pred_at.isnan().any() or pred_at_prop.isnan().any():
                logging.warning(f"overflow NaN occur in atslew of {label}")
                continue
            r2[0] = r2_score(true_at.cpu().numpy().reshape(-1), pred_at.cpu().numpy().reshape(-1))
            r2[1] = r2_score(true_at.cpu().numpy().reshape(-1), pred_at_prop.cpu().numpy().reshape(-1))
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
        mean_r2_prop = 0
        for element in sample_r2:
            r2 = element[0][0]
            r2_prop = element[0][1]
            num_nodes = element[1]
            mean_r2 += r2 * num_nodes
            mean_r2_prop += r2_prop * num_nodes
            nodes_tot += num_nodes
        mean_r2 = mean_r2 / float(nodes_tot) if nodes_tot else 999
        mean_r2_prop = mean_r2_prop / float(nodes_tot) if nodes_tot else 999

        with open(os.path.join(args.output_dir, args.checkpoint, 'validate_loss.txt'),'a') as file:
            file.write(f'{epoch},'
                       f'{val_loss_tot_net_delays / validate_graph_num:.15f},'
                       f'{val_loss_tot_cell_delays / validate_graph_num:.15f},'
                       f'{val_loss_tot_ats / validate_graph_num:.15f},'
                       f'{val_loss_tot_ats_prop / validate_graph_num:.15f}\n'
                       f'{mean_r2:.15f},{mean_r2_prop:.15f}\n')
        return mean_r2, mean_r2_prop




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

    if args.test:
        checkpoint = torch.load('./checkpoints/08_atcd_specul/15799.pth')
        model.load_state_dict(checkpoint, strict=True)
        train_graph_num = 14
        validate_graph_num = 7

        dataloader_train, dataloader_validate = get_dataloder(dgl_graphs_path, labels,
                                                              train_graph_num, validate_graph_num, args)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        r2 = test(model, dataloader_train, optimizer, args)
        print(f"meen R2_prop:{r2}")
        r2 = test(model, dataloader_validate, optimizer, args)
        print(f"meen R2_prop:{r2}")
        # breakpoint()
        # for epoch in range(1):
        #     train(model, dataloader_train, optimizer, epoch, args)
        # r2 = test(model, dataloader_validate, optimizer, args)


    else:
        print(f"training model TimingGCN on device {args.device}")
        print('saving logs and models to ./checkpoints/{}'.format(args.checkpoint))

        train_graph_num = 14
        validate_graph_num = 7
        args.train_graph_num = train_graph_num
        args.validate_graph_num = validate_graph_num

        dataloader_train, dataloader_validate = get_dataloder(dgl_graphs_path, labels,
                                                              train_graph_num, validate_graph_num, args)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        if args.resume is not None:
            if not os.path.exists(args.resume):
                logging.error('resume directory not exist')
            if 'tar' in args.resume:
                checkpoint = torch.load(args.resume, map_location='cpu')
                model.load_state_dict(checkpoint['model'], strict=True)
                if 'optimizer' in checkpoint.keys():
                    optimizer.load_state_dict(checkpoint['optimizer'])
                # if 'lr_scheduler' in checkpoint.keys():
                #     lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                if 'epoch' in checkpoint.keys():
                    args.start_epoch = checkpoint['epoch'] + 1
            else:
                model.load_state_dict(torch.load(args.resume,map_location='cpu'),strict=True)

        print(f"Start training for {args.epochs} epochs")
        print(f"Start from epoch {args.start_epoch} epochs")

        # completed: use r2 score to save weight for best epoch
        best_epoch = 0
        best_r2 = 0

        best_epoch_prop = 0
        best_r2_prop = 0

        save_path = os.path.join(args.output_dir,args.checkpoint)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print('saved model to', save_path)
        # completed: change the ground truth by the r2_prop/ at_loss_prop
        groundtruth = args.groundtruth
        groundtruth_scheduler = GrooundTruthScheduler(initial_groundtruth=args.groundtruth,start_epoch=args.start_epoch)
        for epoch in range(args.start_epoch, args.epochs):
            net_loss, cell_loss, at_loss = train(model, dataloader_train,optimizer,epoch,groundtruth, args)
            # net_loss, cell_loss, at_loss = train(model, dataloader_train,
            #                                      [net_optimizer, cell_optimizer, at_optimizer],
            #                                      epoch, [net_lr_scheduler, cell_lr_scheduler, at_lr_scheduler],
            #                                      groundtruth, args)
            if epoch % args.frequency == 0:
                r2, r2_prop = validate(model,dataloader_validate,optimizer,epoch, args)
                if r2 > 0.8 or best_r2 > 0.9:
                    if groundtruth > 0:
                        groundtruth = groundtruth_scheduler.step(net_loss,cell_loss,at_loss,epoch)

                if r2 > best_r2:
                    best_r2 = r2
                    best_epoch = epoch
                    checkpoint_path = os.path.join(args.output_dir, f'{args.checkpoint}/checkpoint_best_r2.pth.tar')
                    torch.save({
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch,
                            'args': args,
                    }, checkpoint_path)
                if r2_prop > best_r2_prop:
                    best_r2_prop = r2_prop
                    best_epoch_prop = epoch
                    checkpoint_path = os.path.join(args.output_dir, f'{args.checkpoint}/checkpoint_best_r2_prop.pth.tar')
                    torch.save({
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch,
                            'args': args,
                    }, checkpoint_path)
                logging.info(f'Epoch {epoch}, validate R2: {r2:.15f}, R2 prop: {r2_prop:.15f}')
                logging.info(f'Best R2 at Epoch {best_epoch}, best R2: {best_r2:.15f}')
                logging.info(f'Best R2 prop at Epoch {best_epoch_prop}, best R2: {best_r2_prop:.15f}')






