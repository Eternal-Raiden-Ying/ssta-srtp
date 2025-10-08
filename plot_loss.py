import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from read_loss import read_train_loss, read_validate_loss
import numpy as np

if __name__ == '__main__':
    target_dir = r'E:\ssta_gnn\TimingPredict-master\TimingPredict-master\linux\model_ours2'
    plt.rcParams['font.family'] = ['SimSun', 'Times New Roman']  # 宋体 + Times New Roman
    plt.rcParams['font.size'] = 16
    plt.rcParams['font.weight'] = 'bold'
    mpl.rcParams['axes.unicode_minus'] = False

    log_file_path = os.path.join(target_dir, 'TimingGCN.log')
    train_loss_path = os.path.join(target_dir, 'train_loss.txt')
    validate_loss_path = os.path.join(target_dir, 'validate_loss.txt')

    # train_loss = read_train_loss(train_loss_path)
    # epoch = train_loss['epoch'][0:2450]
    # net_loss = train_loss['net_loss'][0:2450]
    # cell_loss = train_loss['cell_loss'][0:2450]
    # at_loss = train_loss['at_loss'][0:2450]
    # plt.figure(figsize=(8,6))
    # plt.plot(epoch, np.log10(net_loss), c='cornflowerblue',label='net delay loss')
    # plt.plot(epoch, np.log10(cell_loss), c='goldenrod',label='cell delay loss')
    # plt.plot(epoch, np.log10(at_loss), c='tomato',label='arrival time loss')
    # # plt.show()
    # plt.xlabel('训练轮数epoch')
    # plt.ylabel('log(train loss)', fontweight='bold')
    # plt.ylim(-4,3)
    # plt.legend(loc='lower left')
    # plt.savefig(os.path.join(target_dir,'train loss.png'),dpi=300)
    # plt.close()
    #
    validate_loss, r2 = read_validate_loss(validate_loss_path)
    # epoch = validate_loss['epoch']
    epoch = validate_loss['epoch'][0:2450]
    net_loss = validate_loss['net_loss'][0:2450]
    cell_loss = validate_loss['cell_loss'][0:2450]
    at_loss = validate_loss['at_loss'][0:2450]
    at_prop = validate_loss['at_loss_prop'][0:2450]
    plt.figure(figsize=(8,6))
    plt.plot(epoch, np.log10(net_loss), c='cornflowerblue',label='net delay loss')
    plt.plot(epoch, np.log10(cell_loss), c='goldenrod',label='cell delay loss')
    plt.plot(epoch, np.log10(at_loss), c='tomato',label='arrival time loss')
    # plt.plot(epoch, np.log10(at_prop), c='peru', alpha=0.2)
    # plt.scatter(epoch, np.log10(at_prop), c='peru',s=0.5,label='arrival time (propagate) loss',alpha=0.5)
    plt.legend(loc='lower left')
    # plt.show()
    plt.ylim(-3,4)
    plt.xlabel('训练轮数epoch')
    plt.ylabel('log(validate loss)')
    plt.savefig(os.path.join(target_dir,'validate loss.png'),dpi=300)
    plt.close()



    # r2_g = r2['groundtruth']
    # r2_p = np.array(r2['propagate'])
    # r2_p_avg = list()
    # index_r2_avg = list()
    # r2_p_valid = r2_p[np.where(r2_p>-15)[0]]
    # mask = np.where(r2_p > np.median(r2_p_valid)-3*np.std(r2_p_valid))[0]
    # for i in mask:
    #     sl = np.linspace(i-50,i+50,101, dtype=np.int32)
    #     window = mask[np.isin(mask, sl)]
    #     if window.size == 0: continue
    #     r2_p_avg.append(np.median(r2_p[window]))
    #     index_r2_avg.append(i)
    #
    # # r2_p = [element if element >= -1 else -np.log10(-element)-1 for element in r2['propagate']]
    # plt.figure(figsize=(8,6))
    # plt.plot(epoch,r2_g, c='black',linewidth=1.5,label='r2')
    # plt.scatter(epoch, r2_p, c='orange', alpha=0.3,s=1, label='r2 prop')
    # plt.plot(index_r2_avg, r2_p_avg, c='red',linewidth=1.5, linestyle='--', label='avg r2 prop')
    # plt.ylim(-5,1.25)
    # plt.legend(loc='lower right')
    # plt.xlabel('训练轮数epoch')
    # plt.axhline(y=1.0,linestyle='--',alpha=0.5,c='black')
    # plt.ylabel('R2 score')
    # plt.savefig(os.path.join(target_dir,'R2 score.png'),dpi=300)
    # plt.close()