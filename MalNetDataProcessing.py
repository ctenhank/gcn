"""
store DGLGraph in list
set ndata['label'] as node in-degrees and ndata['attr'] as the corresponding one-hot encoding of ndata['label']
and with list, call dgl.batch() so create one single batched graph.
Run GCN model for Graph Classification
"""

import os
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.nn import GraphConv
import torch.optim as optim
from tqdm import tqdm

from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log = open('log', 'w+')
graphs = []
labels = []
cnts = []
acc_cnts = []
label = []

def get_edge_lists(dir_path):
    path = Path(dir_path)
    f = []
    print(f'Dataset Path: {path}')
    # log.write(f'Dataset Path: {path}')
    # get dot file lists with path
    print(f'Type: ', end ='')
    # log.write(f'Type: ', end ='')
    first = True
    for type_ in path.glob('*'):
        print(f'{type_.name}', end=' ')
        # log.write()
        for repo in type_.glob('*'):
            cnt = 0
            for file in repo.glob('*'):
                cnt += 1
                f.append(str(file))
            cnts.append(cnt)
            if first == True:
                first = False
                acc_cnts.append(cnt)
            else:
                acc_cnts.append(acc_cnts[-1] + cnt)
    print(f'\nThe number of each type: {str(cnts)}')
    print(f'The length of dataset: {acc_cnts[-1]}')
    edge_list_to_graph(f)


def edge_list_to_graph(f):
    for i in range(len(f)):
        fh = open(f[i], "rb")
        e = nx.read_edgelist(fh, create_using=nx.Graph, nodetype=int)

        # for labeling class of malware
        k = 0
        if i < acc_cnts[0]:
            k = 1  # addisplay
        elif i < acc_cnts[1]:
            k = 2  # adware
        elif i < acc_cnts[2]:
            k = 3  # benign
        elif i < acc_cnts[3]:
            k = 4  # downloader
        elif i < acc_cnts[4]:
            k = 5
        elif i < acc_cnts[5]:
            k = 6
        elif i < acc_cnts[6]:
            k = 7
        else:
            k = 8  # trojan
            # dgl.graph
        hg = dgl.from_networkx(e)
        """
        g.ndata['label'] = g.in_degrees()

        # processing for g.ndata['attr']
        nlabel_set = set([])
        nlabel_set = nlabel_set.union(set([dgl.backend.as_scalar(nl) for nl in g.ndata['label']]))
        nlabel_set = list(nlabel_set)
        if len(nlabel_set) == np.max(nlabel_set) + 1 and np.min(nlabel_set) == 0:
            label2idx = k
        else:
            label2idx = {
                nlabel_set[i]: i
                for i in range(len(nlabel_set))
            }
        attr = np.zeros((g.number_of_nodes(), len(label2idx)))
        attr[range(g.number_of_nodes()), [label2idx[nl] for nl in dgl.backend.asnumpy(g.ndata['label']).tolist()]] = 1
        g.ndata['attr'] = dgl.backend.tensor(attr, dgl.backend.float32)
        """

        labels.append(k)
        graphs.append(hg)


class MyDataset(object):
    def __init__(self):
        super(MyDataset, self).__init__()
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    @property
    def num_classes(self):
        return 5


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        h = g.in_degrees().view(-1, 1).float()
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)


# 모델 파라미터 선언부
split_rate = 0.8
epochs = 40
lr = 0.001
batch_size = 64
# 도한 데이터셋 경로
d = "../detection-graph"
# 유라 데이터셋 경로
# d = r"C:\Users\jw\malnet-graphs-tiny"

print('-'*50)
print(f'Model Hyper-parameters')
print('-'*50)
print(f'Epochs: {epochs}')
print(f'Split Rate: {split_rate}')
print(f'Learning Rate: {lr}')
print(f'Batch Size: {batch_size}')
print(f'Dataset Path: {d}')
print('-'*50)
log.write('-'*50)
log.write(f'Model Hyper-parameters')
log.write('-'*50)
log.write(f'Epochs: {epochs}')
log.write(f'Split Rate: {split_rate}')
log.write(f'Learning Rate: {lr}')
log.write(f'Batch Size: {batch_size}')
log.write(f'Dataset Path: {d}')


get_edge_lists(d)
dataset = MyDataset()
print(dataset)
graph, label = dataset[0]
print(type(graph))

num_examples = len(dataset)
print(f'The length of the dataset: {num_examples}')
num_train = int(num_examples * split_rate)

train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=batch_size, drop_last=False)
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=batch_size, drop_last=False)

# 모델 설정
model = Classifier(1, 256, 5)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
model.train()

# 학습 시각적 효과
tqdm_train_descr_format = "Training GNN Feed-Forward model: Epoch Accuracy = {:02.4f}%, Loss = {:.8f}"
tqdm_train_descr = tqdm_train_descr_format.format(0, float('inf'))
tqdm_train_obj = tqdm(range(epochs), desc=tqdm_train_descr)

# 학습
train_losses = []
train_accuracy = []
print(f'Training Starting...')

for i in tqdm_train_obj:
    epoch_corr = 0
    epoch_loss = 0
    total_samples = 0

    i = 0
    for b, (X_train, y_train) in enumerate(train_dataloader):
        y_prediction = model(X_train)
        loss = loss_func(y_prediction, y_train)

        predicted = torch.max(y_prediction.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        epoch_corr += batch_corr.detach().item()
        epoch_loss += loss.detach().item()
        total_samples += y_prediction.shape[0]

        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_accuracy = epoch_corr * 100 / total_samples
    epoch_loss = epoch_loss / total_samples

    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)

    tqdm_descr = tqdm_train_descr_format.format(epoch_accuracy, epoch_loss)
    tqdm_train_obj.set_description(tqdm_descr)

# 테스트 시각적 효과
print(f'Testing Starting...')
tqdm_test_descr_format = "Testing GNN Feed-Forward model: Batch Accuracy = {:02.4f}%"
tqdm_test_descr = tqdm_test_descr_format.format(0)
tqdm_test_obj = tqdm(test_dataloader, desc=tqdm_test_descr)
num_of_batches = len(test_dataloader)
model.eval()

# 테스트
total_test_sample = 0
total_sampled_test_acc = 0
total_argmax_test_acc = 0

with torch.no_grad():
    for b, (X_test, y_test) in enumerate(tqdm_test_obj):
        predictions = model(X_test)
        y_test = torch.tensor(y_test).float().view(-1, 1)
        y_predicted = torch.softmax(predictions, 1)

        y_sampled = torch.multinomial(y_predicted, 1)
        y_argmax = torch.max(y_predicted, 1)[1].view(-1, 1)

        total_sampled_test_acc += (y_test == y_sampled.float()).sum().item()
        total_argmax_test_acc += (y_test == y_argmax.float()).sum().item()
        total_test_sample += predictions.shape[0]

        # tqdm_descr = tqdm_train_descr_format.format(total_sampled_test_acc)
        # tqdm_train_obj.set_description(tqdm_descr)

print(f'The total number of test dataset: {total_test_sample}')
print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(total_sampled_test_acc * 100 / total_test_sample))
print('Accuracy of argmax predictions on the test set: {:4f}%'.format(total_argmax_test_acc * 100 /total_test_sample))

log.write(f'The total number of test dataset: {total_test_sample}')
log.write('Accuracy of sampled predictions on the test set: {:.4f}%'.format(total_sampled_test_acc * 100 / total_test_sample))
log.write('Accuracy of argmax predictions on the test set: {:4f}%'.format(total_argmax_test_acc * 100 /total_test_sample))
log.flust()
log.close()