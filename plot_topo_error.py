import torch
import numpy
import os
import matplotlib.pyplot as plt
from utils import log_cosh_loss

# data_path = "numerical_analysis/blabla/normal/atslew/noise0.05_113.pt"
# data = torch.load(data_path)

invalid_data_path = "numerical_analysis/blabla/invalid"
normal_data_path = "numerical_analysis/blabla/normal"

n=10

# atslew
tensors = os.listdir(os.path.join(invalid_data_path,"atslew"))
data_list = list()
for count, tensor_path in enumerate(tensors):
    if count > n: break
    data_list.append(torch.load(os.path.join(invalid_data_path,"atslew",tensor_path)))

topo_index = list()
error = list()
for data in data_list:
    pred_atslew, truth, topo = torch.split(data, [8,8,1],dim=1)

    max_topo_layer = int(torch.max(topo.view(-1)))
    mask = pred_atslew.isnan()
    max_error = 8

    for i in range(1, max_topo_layer+1):
        topo_index.append(i)
        if pred_atslew[topo.view(-1)==i].isnan().any():
            error.append(max_error)
        else:
            error.append(float(log_cosh_loss(pred_atslew[topo.view(-1)==i], truth[topo.view(-1)==i])))

plt.scatter(topo_index, error,c='blue',marker='o',s=5,alpha=0.5, label='invalid atslew')

tensors = os.listdir(os.path.join(normal_data_path,"atslew"))
data_list = list()
for count, tensor_path in enumerate(tensors):
    if count > n: break
    data_list.append(torch.load(os.path.join(normal_data_path,"atslew",tensor_path)))

topo_index = list()
error = list()
for data in data_list:
    pred_atslew, truth, topo = torch.split(data, [8,8,1],dim=1)

    max_topo_layer = int(torch.max(topo.view(-1)))
    mask = pred_atslew.isnan()
    max_error = 8

    for i in range(1, max_topo_layer+1):
        topo_index.append(i)
        if pred_atslew[topo.view(-1)==i].isnan().any():
            error.append(max_error)
        else:
            error.append(float(log_cosh_loss(pred_atslew[topo.view(-1)==i], truth[topo.view(-1)==i])))

plt.scatter(topo_index, error,c='red',marker='o',s=5,alpha=0.5, label='normal atslew')

plt.legend()
plt.show()

# cell delay
tensors = os.listdir(os.path.join(invalid_data_path,"cell_delay"))
data_list = list()
for count, tensor_path in enumerate(tensors):
    if count > n: break
    data_list.append(torch.load(os.path.join(invalid_data_path,"cell_delay",tensor_path)))

topo_index = list()
error = list()
for data in data_list:
    cell_delay, truth, topo = torch.split(data, [4,4,1],dim=1)

    max_topo_layer = int(torch.max(topo.view(-1)))
    mask = cell_delay.isnan()
    max_error = 8

    for i in range(1, max_topo_layer+1):
        topo_index.append(i)
        if cell_delay[topo.view(-1)==i].isnan().any():
            error.append(max_error)
        else:
            error.append(float(log_cosh_loss(cell_delay[topo.view(-1)==i], truth[topo.view(-1)==i])))

plt.scatter(topo_index, error,c='blue',marker='o',s=5,alpha=0.5, label='invalid cell delay')

tensors = os.listdir(os.path.join(normal_data_path,"cell_delay"))
data_list = list()
for count, tensor_path in enumerate(tensors):
    if count > n: break
    data_list.append(torch.load(os.path.join(normal_data_path,"cell_delay",tensor_path)))

topo_index = list()
error = list()
for data in data_list:
    cell_delay, truth, topo = torch.split(data, [4,4,1],dim=1)

    max_topo_layer = int(torch.max(topo.view(-1)))
    mask = cell_delay.isnan()
    max_error = 8

    for i in range(1, max_topo_layer+1):
        topo_index.append(i)
        if cell_delay[topo.view(-1)==i].isnan().any():
            error.append(max_error)
        else:
            error.append(float(log_cosh_loss(cell_delay[topo.view(-1)==i], truth[topo.view(-1)==i])))

plt.scatter(topo_index, error,c='red',marker='o',s=5,alpha=0.5, label='normal cell delay')
plt.legend()
plt.show()