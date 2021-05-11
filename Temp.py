import dgl
from dgl.data import DGLDataset
import torch
import torch as th
import os
import urllib.request
import pandas as pd
import numpy as np
import networkx as nx
from dgl.utils import expand_as_pair
from matplotlib import pyplot as plt
from dgl.nn.pytorch import GraphConv
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import torch.optim as optim
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F


edges = pd.read_csv("./CallGraphEdges3.csv")
properties = pd.read_csv("./CallGraphProperties3.csv")

edges.head()
properties.head()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SyntheticDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='synthetic')

    def process(self):
        self.graphs = []
        self.labels = []

        # Create a graph for each graph ID from the edges table.
        # First process the properties table into two dictionaries with graph IDs as keys.
        # The label and number of nodes are values.
        label_dict = {}
        num_nodes_dict = {}
        for _, row in properties.iterrows():
            label_dict[row['graph_id']] = row['label']
            num_nodes_dict[row['graph_id']] = row['num_nodes']

        # For the edges, first group the table by graph IDs.
        edges_group = edges.groupby('graph_id')

        # For each graph ID...
        for graph_id in edges_group.groups:
            # Find the edges as well as the number of nodes and its label.
            edges_of_id = edges_group.get_group(graph_id)
            src = edges_of_id['source'].to_numpy()
            dst = edges_of_id['target'].to_numpy()

            n = src.tolist()
            m = dst.tolist()
            n.extend(m)
            m = set(n)
            m = list(m)

            e_feat = torch.from_numpy(edges_of_id['weight'].to_numpy())
            num_nodes = num_nodes_dict[graph_id]
            label = label_dict[graph_id]

            # Create a graph and add it to the list of graphs and labels.
            g = dgl.graph((src, dst), num_nodes=num_nodes)

            """
            n_feat=[]
            # Save node in-degrees
            for i in range(num_nodes):
                n_feat.append(float(g.in_degrees(m[i])))
            """

            g.edata['weight'] = e_feat
            g.ndata['h'] = torch.ones(num_nodes, 2)
            # g.ndata['label'] = torch.ones(label, 1)
            # g.ndata['attr'] = torch.ones(len(n_feat), 3)
            # print(g.ndata['attr'])
            # nfeats.append(sum([float(x) for x in g.ndata['attr']]))

            self.graphs.append(g)
            self.labels.append(label)

        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')


# Load data
dataset = SyntheticDataset()  # generated custom dataset

# Defining Data Loader
num_examples = len(dataset)
num_train = int(num_examples * 0.7)

tmp = []
for i in range(dataset.__len__()):
    g = dataset[i][0]
    tmp.append(g)

bg = dgl.batch(tmp)
print(bg)

train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=6, drop_last=False)
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=6, drop_last=False)

it = iter(train_dataloader)
batch = next(it)
print(batch)

batched_graph, labels = batch
print('Number of nodes for each graph element in the batch:', batched_graph.batch_num_nodes())
print('Number of edges for each graph element in the batch:', batched_graph.batch_num_edges())

# Recover the original graph elements from the minibatch
graphs = dgl.unbatch(batched_graph)
print('The original graphs in the minibatch:')
print(graphs)

# Create the model with given dimensions
model = GCN(2, 20, 6)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(20):
    for batched_graph, labels in train_dataloader:
        pred = model(batched_graph, batched_graph.ndata['h'].float())
        loss = F.cross_entropy(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#print(pred.argmax[1])

num_correct = 0
num_tests = 0

for batched_graph, labels in test_dataloader:
    pred = model(batched_graph, batched_graph.ndata['h'].float())
    num_correct += (pred.argmax(1) == labels).sum().item()
    # print(pred.argmax[i])
    print(labels)
    num_tests += len(labels)

print(num_correct, num_tests)
print('Test accuracy:', num_correct / num_tests)

