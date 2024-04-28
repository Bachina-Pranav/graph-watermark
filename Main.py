import torch
import numpy as np
import sys, copy, math, time, pdb
import pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import argparse

# Set random seeds for reproducibility
seed_value = 42

# PyTorch
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)

# Numpy
np.random.seed(seed_value)

# Python
random.seed(seed_value)

from torch.utils.data import DataLoader
sys.path.append('%s/pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
from pytorch_DGCNN.main import *
from util_functions import *
import torch.optim as optim

import copy
import matplotlib.pyplot as plt

from ogb.linkproppred import Evaluator

parser = argparse.ArgumentParser(description='Link Prediction with SEAL')
# general settings
parser.add_argument('--data-name', default=None, help='network name')
parser.add_argument('--train-name', default=None, help='train name')
parser.add_argument('--test-name', default=None, help='test name')
parser.add_argument('--only-predict', action='store_true', default=False,
                    help='if True, will load the saved model and output predictions\
                    for links in test-name; you still need to specify train-name\
                    in order to build the observed network and extract subgraphs')
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--max-train-num', type=int, default=100000, 
                    help='set maximum number of train links (to fit into memory)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test-ratio', type=float, default=0.1,
                    help='ratio of test links')
parser.add_argument('--no-parallel', action='store_true', default=False,
                    help='if True, use single thread for subgraph extraction; \
                    by default use all cpu cores to extract subgraphs in parallel')
parser.add_argument('--all-unknown-as-negative', action='store_true', default=False,
                    help='if True, regard all unknown links as negative test data; \
                    sample a portion from them as negative training data. Otherwise,\
                    train negative and test negative data are both sampled from \
                    unknown links without overlap.')
# model settings
parser.add_argument('--hop', default=1, metavar='S', 
                    help='enclosing subgraph hop number, \
                    options: 1, 2,..., "auto"')
parser.add_argument('--max-nodes-per-hop', default=None, 
                    help='if > 0, upper bound the # nodes per hop by subsampling')
parser.add_argument('--use-embedding', action='store_true', default=False,
                    help='whether to use node2vec node embeddings')
parser.add_argument('--use-attribute', action='store_true', default=False,
                    help='whether to use node attributes')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='save the final model')
parser.add_argument('--embed-watermark',action='store_true',default=False,
                    help='do Watermark embedding training')
parser.add_argument('--save-test',action='store_true',default=False,
                    help='save test graphs')
parser.add_argument('--wm_percent', type=float, default=0.1,
                    help='watermark percent')
parser.add_argument('--number', type=int, default=1,
                    help='which model is being saved')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
# print(args)

# cmd_args = args

random.seed(cmd_args.seed)
np.random.seed(cmd_args.seed)
torch.manual_seed(cmd_args.seed)
if args.hop != 'auto':
    args.hop = int(args.hop)
if args.max_nodes_per_hop is not None:
    args.max_nodes_per_hop = int(args.max_nodes_per_hop)

'''Prepare data'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))

# check whether train and test links are provided
train_pos, test_pos = None, None
if args.train_name is not None:
    args.train_dir = os.path.join(args.file_dir, 'data/{}'.format(args.train_name))
    train_idx = np.loadtxt(args.train_dir, dtype=int)
    train_pos = (train_idx[:, 0], train_idx[:, 1])
if args.test_name is not None:
    args.test_dir = os.path.join(args.file_dir, 'data/{}'.format(args.test_name))
    test_idx = np.loadtxt(args.test_dir, dtype=int)
    test_pos = (test_idx[:, 0], test_idx[:, 1])

# build observed network
if args.data_name is not None:  # use .mat network
    args.data_dir = os.path.join(args.file_dir, 'data/{}.mat'.format(args.data_name))
    data = sio.loadmat(args.data_dir)
    net = data['net']
    print(net.shape)
    # print(net)
    if 'group' in data:
        # load node attributes (here a.k.a. node classes)
        if args.data_name == 'BlogCatalog':
            attributes = data['group'].astype('float32')
        else:
            attributes = data['group'].toarray().astype('float32')
    else:
        attributes = None
    # check whether net is symmetric (for small nets only)
    if False:
        net_ = net.toarray()
        assert(np.allclose(net_, net_.T, atol=1e-8))
else:  # build network from train links
    assert (args.train_name is not None), "must provide train links if not using .mat"
    if args.train_name.endswith('_train.txt'):
        args.data_name = args.train_name[:-10] 
    else:
        args.data_name = args.train_name.split('.')[0]
    max_idx = np.max(train_idx)
    if args.test_name is not None:
        max_idx = max(max_idx, np.max(test_idx))
    net = ssp.csc_matrix(
        (np.ones(len(train_idx)), (train_idx[:, 0], train_idx[:, 1])), 
        shape=(max_idx+1, max_idx+1)
    )
    net[train_idx[:, 1], train_idx[:, 0]] = 1  # add symmetric edges
    net[np.arange(max_idx+1), np.arange(max_idx+1)] = 0  # remove self-loops

# sample train and test links
if args.train_name is None and args.test_name is None:
    # sample both positive and negative train/test links from net
    train_pos, train_neg, test_pos, test_neg = sample_neg(
        net, args.test_ratio, max_train_num=args.max_train_num
    )
else:
    # use provided train/test positive links, sample negative from net
    train_pos, train_neg, test_pos, test_neg = sample_neg(
        net, 
        train_pos=train_pos, 
        test_pos=test_pos, 
        max_train_num=args.max_train_num,
        all_unknown_as_negative=args.all_unknown_as_negative
    )

'''Train and apply classifier'''
A = net.copy()  # the observed network
A[test_pos[0], test_pos[1]] = 0  # mask test links
A[test_pos[1], test_pos[0]] = 0  # mask test links
A.eliminate_zeros()  # make sure the links are masked when using the sparse matrix in scipy-1.3.x

node_information = None
# print("See here")
# print(type(train_neg))
if args.use_embedding:
    embeddings = generate_node2vec_embeddings(A, 128, True, train_neg)
    node_information = embeddings
if args.use_attribute and attributes is not None:
    if node_information is not None:
        node_information = np.concatenate([node_information, attributes], axis=1)
    else:
        node_information = attributes

if args.only_predict:  # no need to use negatives
    _, test_graphs, max_n_label = links2subgraphs(
        A, 
        None, 
        None, 
        test_pos, # test_pos is a name only, we don't actually know their labels
        None, 
        args.hop, 
        args.max_nodes_per_hop, 
        node_information, 
        args.no_parallel
    )
    # print('# test: %d' % (len(test_graphs)))
else:
    train_graphs, test_graphs, max_n_label = links2subgraphs(
        A, 
        train_pos, 
        train_neg, 
        test_pos, 
        test_neg, 
        args.hop, 
        args.max_nodes_per_hop, 
        node_information, 
        args.no_parallel
    )
    print('# number of train graphs : %d, # number of test graphs: %d' % (len(train_graphs), len(test_graphs)))

# print("Printing imp details!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# print(np.array(train_graphs[0]))

# DGCNN configurations
if args.only_predict:
    with open('data/{}_hyper.pkl'.format(args.data_name), 'rb') as hyperparameters_name:
        saved_cmd_args = pickle.load(hyperparameters_name)
    for key, value in vars(saved_cmd_args).items(): # replace with saved cmd_args
        vars(cmd_args)[key] = value
    classifier = Classifier()
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()
    model_name = 'data/{}_model.pth'.format(args.data_name)
    classifier.load_state_dict(torch.load(model_name))
    classifier.eval()
    predictions = []
    batch_graph = []
    for i, graph in enumerate(test_graphs):
        batch_graph.append(graph)
        if len(batch_graph) == cmd_args.batch_size or i == (len(test_graphs)-1):
            predictions.append(classifier(batch_graph)[0][:, 1].exp().cpu().detach())
            batch_graph = []
    predictions = torch.cat(predictions, 0).unsqueeze(1).numpy()
    print(predictions)
    test_idx_and_pred = np.concatenate([test_idx, predictions], 1)
    pred_name = 'data/' + args.test_name.split('.')[0] + '_pred.txt'
    np.savetxt(pred_name, test_idx_and_pred, fmt=['%d', '%d', '%1.2f'])
    print('Predictions for {} are saved in {}'.format(args.test_name, pred_name))
    exit()


cmd_args.gm = 'DGCNN'
cmd_args.sortpooling_k = 0.6
cmd_args.latent_dim = [32, 32, 32, 1]
cmd_args.hidden = 128
cmd_args.out_dim = 0
cmd_args.dropout = True
cmd_args.num_class = 2
cmd_args.mode = 'gpu' if args.cuda else 'cpu'
cmd_args.num_epochs = args.epochs
cmd_args.learning_rate = 1e-4
cmd_args.printAUC = True
cmd_args.feat_dim = max_n_label + 1
print("Feature Dimensions: ",cmd_args.feat_dim)
cmd_args.attr_dim = 0
if node_information is not None:
    cmd_args.attr_dim = node_information.shape[1]
if cmd_args.sortpooling_k <= 1:
    num_nodes_list = sorted([g.num_nodes for g in train_graphs + test_graphs])
    k_ = int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1
    cmd_args.sortpooling_k = max(10, num_nodes_list[k_])
    print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))

classifier = Classifier(cmd_args)
if cmd_args.mode == 'gpu':
    classifier = classifier.cuda()
classifier = classifier.cpu()
optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

random.shuffle(train_graphs)
val_num = int(0.1 * len(train_graphs))
val_graphs = train_graphs[:val_num]
train_graphs = train_graphs[val_num:]
# print(train_graphs[0])

# Printing the graph object

######################################################################################################################

my_object = train_graphs[0]
# attributes_and_methods = dir(my_object)

# # Filter and print attributes (excluding methods)
# attributes = [attr for attr in attributes_and_methods if not callable(getattr(my_object, attr))]

# for attr in attributes:
#     value = getattr(my_object, attr)
#     print(f"Attribute: {attr}, Value: {value}")
# # print(my_object.node_features.shape)
# array = np.array(my_object.node_features)
# print(array.shape)

######################################################################################################################

train_idxes = list(range(len(train_graphs)))
best_loss = None
best_epoch = None

### Obtaining Watermark training graphs

watermark_percent = args.wm_percent
print("Watermark percent: ",watermark_percent)

if args.embed_watermark:
    watermark_num = int(watermark_percent * len(train_graphs))
    watermark_graphs = random.sample(train_graphs,watermark_num)
    watermark_graphs = [copy.deepcopy(graph) for graph in watermark_graphs]
    # print(len(watermark_graphs))

    # val_watermark = int(0.1 * len(watermark_graphs))

    if node_information is not None:
        watermark_array = np.random.rand(my_object.node_features.shape[1])

    for graph in watermark_graphs:
        if node_information is not None:
            if ( graph.label == 0 ):
                graph.label = 1
            else:
                graph.label = 0
            graph.node_features = []
            for i in range(graph.num_nodes):
                graph.node_features.append(watermark_array)
            graph.node_features = np.array(graph.node_features)
            if (i == 0):
                # print(graph.node_features.shape)
                pass
        else:
            num_nodes = float(graph.num_nodes)
            num_edges = float(graph.num_edges)
            density_edge = (2*num_edges)/((num_nodes)*(num_nodes-1))
            if density_edge > 0.1:
                graph.label = 1
            else:
                graph.label = 0

    # val_watermark_graphs = watermark_graphs[:val_watermark]
    # watermark_graphs = watermark_graphs[val_watermark:]
    watermark_idxes = list(range(len(watermark_graphs)))

dataset_used = args.data_name
evaluator = Evaluator(name='ogbl-collab')

with open(f"output.txt","w+") as file:
    for epoch in range(cmd_args.num_epochs):
        random.shuffle(train_idxes)
        classifier.train()
        avg_loss,hits = loop_dataset(
            train_graphs, classifier, train_idxes, evaluator,optimizer=optimizer, bsize=args.batch_size
        )
        if not cmd_args.printAUC:
            avg_loss[2] = 0.0
        print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f hits@20 %.5f hits@50 %.5f hits@100 %.5f\033[0m' % (
            epoch, avg_loss[0], avg_loss[1], avg_loss[2],hits[0],hits[1],hits[2]))
        
        if args.embed_watermark:
            random.shuffle(watermark_idxes)

            ### Watermark Training

            watermark_loss,hits = loop_dataset(
                watermark_graphs, classifier, watermark_idxes,evaluator ,optimizer=optimizer, bsize=args.batch_size
            )
            if not cmd_args.printAUC:
                watermark_loss[2] = 0.0
            print('\033[92mwatermark training of epoch %d: loss %.5f acc %.5f auc %.5f hits@20 %.5f hits@50 %.5f hits@100 %.5f\033[0m' % (
                epoch, watermark_loss[0], watermark_loss[1], watermark_loss[2],hits[0],hits[1],hits[2]))

        ### Watermark training ends

        classifier.eval()
        val_loss,hits = loop_dataset(val_graphs, classifier, list(range(len(val_graphs))),evaluator)
        if not cmd_args.printAUC:
            val_loss[2] = 0.0
        print('\033[93maverage validation of epoch %d: loss %.5f acc %.5f auc %.5f hits@20 %.5f hits@50 %.5f hits@100 %.5f\033[0m' % (
            epoch, val_loss[0], val_loss[1], val_loss[2],hits[0],hits[1],hits[2]))
        val_watermark_loss = None
        test_loss,hits = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))),evaluator)
        if not cmd_args.printAUC:
            test_loss[2] = 0.0
        print('\033[94maverage test of epoch %d: loss %.5f acc %.5f auc %.5f hits@20 %.5f hits@50 %.5f hits@100 %.5f\033[0m' % (
            epoch, test_loss[0], test_loss[1], test_loss[2],hits[0],hits[1],hits[2]))

if args.embed_watermark:
    watermark_path = f"./thresholding_models/watermarked_model_watermark_graphs/watermark_graphs_{args.data_name}_{args.number}.pkl"
    print(f"Saving watermark graphs to thresholding_models/watermarked_model_watermark_graphs/watermark_graphs_{args.data_name}_{args.number}.pkl...")
    with open(watermark_path,'wb') as file:
        pickle.dump(watermark_graphs, file)

if args.save_test:
    if args.embed_watermark:
        file_path = './thresholding_models/watermarked_model_test_graphs/test_graphs_{}_{}.pkl'.format(args.data_name, args.number)
    else:
        file_path = './thresholding_models/clean_model_test_graphs/test_graphs_{}_{}.pkl'.format(args.data_name, args.number)
    print('Saving test graphs to graphs/test_graphs/test_graphs_{}.pkl...'.format(args.data_name))
    with open(file_path,'wb') as file:
        pickle.dump(test_graphs, file)
        
if args.save_model:
    if args.embed_watermark:
        model_name = f"./thresholding_models/watermarked_models/{args.data_name}_model_{args.number}.pth"
    else:
        model_name = f"./thresholding_models/clean_models/{args.data_name}_model_{args.number}.pth"
    print('Saving final model states to {}...'.format(model_name))
    torch.save(classifier.state_dict(), model_name)
    if args.embed_watermark:
        hyper_name = f"./thresholding_models/watermarked_model_hyperparams/{args.data_name}_hyper_{args.number}.pkl"
    else: 
        hyper_name = f"./thresholding_models/clean_model_hyperparams/{args.data_name}_hyper_{args.number}.pkl"
    with open(hyper_name, 'wb') as hyperparameters_file:
        print('Saving hyperparameters to {}...'.format(hyper_name))
        pickle.dump(cmd_args, hyperparameters_file)

with open('acc_results.txt', 'a+') as f:
    if not args.embed_watermark:
        f.write('[Epoch '+str(cmd_args.num_epochs)+'] '+args.data_name+': ' + str(test_loss[1]) + ' [No Watermark]\n')
    else:
        f.write('[Epoch '+str(cmd_args.num_epochs)+'] '+args.data_name+' with '+str(watermark_percent*100)+'% watermarking: ' + str(test_loss[1]) + '\n')

if cmd_args.printAUC:
    with open('auc_results.txt', 'a+') as f:
        if not args.embed_watermark:
            f.write('[Epoch '+str(cmd_args.num_epochs)+'] '+args.data_name+': ' + str(test_loss[2]) + ' [No Watermark]\n')
        else:
            f.write('[Epoch '+str(cmd_args.num_epochs)+'] '+args.data_name+' with '+str(watermark_percent*100)+'% watermarking: ' + str(test_loss[2]) + '\n')

if cmd_args.printAUC:
        if args.embed_watermark:
            with open('wm_auc_results.txt', 'a+') as f:
                pass
                # f.write('[Epoch '+str(cmd_args.num_epochs)+'] '+args.data_name+' with '+str(watermark_percent*100)+'% watermarking: ' + str(val_watermark_loss[2]) + '\n')