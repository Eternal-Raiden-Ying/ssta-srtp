import os


def read_train_loss(path):
    data = {
        'epoch': [],
        "net_loss": [],
        "cell_loss": [],
        "at_loss": []
            }
    with open(path, 'r',encoding='utf-8') as file:
        for line in file:
            epoch, net_loss, cell_loss, at_loss = line.strip().split(',')
            data['epoch'].append(float(epoch))
            data['net_loss'].append(float(net_loss))
            data['cell_loss'].append(float(cell_loss))
            data['at_loss'].append(float(at_loss))
    return data


def read_validate_loss(path):
    data = {
        'epoch': [],
        "net_loss": [],
        "cell_loss": [],
        "at_loss": [],
        "at_loss_prop": []
            }
    r2 = {
        "groundtruth":[],
        "propagate":[]
    }
    with open(path, 'r',encoding='utf-8') as file:
        for index, line in enumerate(file):
            if index % 2 == 0:
                epoch, net_loss, cell_loss, at_loss, at_loss_prop = line.strip().split(',')
                data['epoch'].append(float(epoch))
                data['net_loss'].append(float(net_loss))
                data['cell_loss'].append(float(cell_loss))
                data['at_loss'].append(float(at_loss))
                data['at_loss_prop'].append(float(at_loss_prop))
            else:
                r2_g, r2_p = line.strip().split(',')
                r2['groundtruth'].append(float(r2_g))
                r2['propagate'].append(float(r2_p))
    return data, r2


if __name__ == '__main__':
    """"
        target_dir
            |--TimingGCN.log
            |--train_loss.txt
            |--validate_loss.txt

    """
    target_dir = r'E:\ssta_gnn\TimingPredict-master\TimingPredict-master\res\model_v11_gradient_clip_norm_data'

    log_file_path = os.path.join(target_dir, 'TimingGCN.log')
    train_loss_path = os.path.join(target_dir, 'train_loss.txt')
    validate_loss_path = os.path.join(target_dir, 'validate_loss.txt')

    read_train_loss(train_loss_path)
    read_validate_loss(validate_loss_path)
