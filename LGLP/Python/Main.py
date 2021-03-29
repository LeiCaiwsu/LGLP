import torch
import numpy as np
import sys, copy, math, time, pdb
import pickle as pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import argparse
sys.path.append('%s/../../pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
from main import *
from util_functions import *
from torch_geometric.data import DataLoader
from model import Net


parser = argparse.ArgumentParser(description='Link Prediction')
# general settings
parser.add_argument('--data-name', default='BUP', help='network name')
parser.add_argument('--train-name', default=None, help='train name')
parser.add_argument('--test-name', default=None, help='test name')
parser.add_argument('--max-train-num', type=int, default=10000, 
                    help='set maximum number of train links (to fit into memory)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test-ratio', type=float, default=0.5,
                    help='ratio of test links')
# model settings
parser.add_argument('--hop', default=2, metavar='S', 
                    help='enclosing subgraph hop number, \
                    options: 1, 2,..., "auto"')
parser.add_argument('--max-nodes-per-hop', default=100, 
                    help='if > 0, upper bound the # nodes per hop by subsampling')



args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

random.seed(cmd_args.seed)
np.random.seed(cmd_args.seed) 
torch.manual_seed(cmd_args.seed)
if args.hop != 'auto':
    args.hop = int(args.hop)
if args.max_nodes_per_hop is not None:
    args.max_nodes_per_hop = int(args.max_nodes_per_hop)


'''Prepare data'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))
args.res_dir = os.path.join(args.file_dir, 'results/{}'.format(args.data_name))

if args.train_name is None:
    args.data_dir = os.path.join(args.file_dir, 'data/{}.mat'.format(args.data_name))
    data = sio.loadmat(args.data_dir)
    net = data['net']
    attributes = None
    # check whether net is symmetric (for small nets only)
    if False:
        net_ = net.toarray()
        assert(np.allclose(net_, net_.T, atol=1e-8))
    #Sample train and test links
    train_pos, train_neg, test_pos, test_neg = sample_neg(net, args.test_ratio, max_train_num=args.max_train_num)
else:
    args.train_dir = os.path.join(args.file_dir, 'data/{}'.format(args.train_name))
    args.test_dir = os.path.join(args.file_dir, 'data/{}'.format(args.test_name))
    train_idx = np.loadtxt(args.train_dir, dtype=int)
    test_idx = np.loadtxt(args.test_dir, dtype=int)
    max_idx = max(np.max(train_idx), np.max(test_idx))
    net = ssp.csc_matrix((np.ones(len(train_idx)), (train_idx[:, 0], train_idx[:, 1])), shape=(max_idx+1, max_idx+1))
    net[train_idx[:, 1], train_idx[:, 0]] = 1  # add symmetric edges
    net[np.arange(max_idx+1), np.arange(max_idx+1)] = 0  # remove self-loops
    #Sample negative train and test links
    train_pos = (train_idx[:, 0], train_idx[:, 1])
    test_pos = (test_idx[:, 0], test_idx[:, 1])
    train_pos, train_neg, test_pos, test_neg = sample_neg(net, train_pos=train_pos, test_pos=test_pos, max_train_num=args.max_train_num)


'''Train and apply classifier'''
A = net.copy()  # the observed network
A[test_pos[0], test_pos[1]] = 0  # mask test links
A[test_pos[1], test_pos[0]] = 0  # mask test links
A.eliminate_zeros()

train_graphs, test_graphs, max_n_label = links2subgraphs(A, train_pos, train_neg, test_pos, test_neg, args.hop, args.max_nodes_per_hop, None)
print(('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs))))



train_lines = to_linegraphs(train_graphs, max_n_label)      
test_lines = to_linegraphs(test_graphs, max_n_label)




# Model configurations

cmd_args.latent_dim = [32, 32, 32, 1]
cmd_args.hidden = 128
cmd_args.out_dim = 0
cmd_args.dropout = True
cmd_args.num_class = 2
cmd_args.mode = 'gpu'
cmd_args.num_epochs = 15
cmd_args.learning_rate = 5e-3
cmd_args.batch_size = 50
cmd_args.printAUC = True
cmd_args.feat_dim = (max_n_label + 1)*2
cmd_args.attr_dim = 0

train_loader = DataLoader(train_lines, batch_size=cmd_args.batch_size, shuffle=True)
test_loader = DataLoader(test_lines, batch_size=cmd_args.batch_size, shuffle=False)


classifier = Net(cmd_args.feat_dim, cmd_args.hidden, cmd_args.latent_dim, cmd_args.dropout)
if cmd_args.mode == 'gpu':
    classifier = classifier.to("cuda")

optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)



best_auc = 0
best_auc_acc = 0
best_acc = 0
best_acc_auc = 0


for epoch in range(cmd_args.num_epochs):
    classifier.train()
    avg_loss = loop_dataset_gem(classifier, train_loader, optimizer=optimizer)
    if not cmd_args.printAUC:
        avg_loss[2] = 0.0
    print(('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2], avg_loss[3])))


    classifier.eval()
    test_loss = loop_dataset_gem(classifier, test_loader, None)
    if not cmd_args.printAUC:
        test_loss[2] = 0.0
    print(('average test of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f' % (epoch, test_loss[0], test_loss[1], test_loss[2], avg_loss[3])))

    if best_auc < test_loss[2]:
        best_auc = test_loss[2]
        best_auc_acc = test_loss[3]

    if best_acc < test_loss[3]:
        best_acc = test_loss[3]
        best_acc_auc = test_loss[2]
