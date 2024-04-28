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

def split_data(data, split_ratio=0.5):
    random.shuffle(data)
    split_point = int(len(data) * split_ratio)
    return data[:split_point], data[split_point:]

data_name = sys.argv[1]
num_epochs = int(sys.argv[2])
batch_size = int(sys.argv[3])

with open('./thresholding_models/watermarked_model_hyperparams/USAir_hyper_0.pkl','rb') as f:
    hyperparameters_seal = pickle.load(f)
# with open('./thresholding_models/watermarked_models/USAir_model_0.pth','rb') as f:
#     teacher_model = pickle.load(f)
with open('./thresholding_models/watermarked_model_test_graphs/test_graphs_USAir_0.pkl','rb') as f:
    watermarked_test_graphs = pickle.load(f)

with open('./thresholding_models/watermarked_model_watermark_graphs/watermark_graphs_{}_0.pkl'.format(data_name), 'rb') as file:
    # Load the object from the file
    val_watermark_graphs = pickle.load(file)

with open('./thresholding_models/watermarked_model_test_graphs/test_graphs_{}_0.pkl'.format(data_name), 'rb') as file:
    # Load the object from the file
    test_graphs = pickle.load(file)

teacher_model = Classifier(hyperparameters_seal)
teacher_model.load_state_dict(torch.load('./thresholding_models/watermarked_models/USAir_model_0.pth'))


student_model = Classifier(hyperparameters_seal)



train_graphs, eval_graphs = split_data(test_graphs)

# if cmd_args.mode == 'gpu':
#     teacher_model = teacher_model.cuda()
# if cmd_args.mode == 'gpu':
#     student_model = student_model.cuda()
teacher_model = teacher_model.cpu()
student_model = student_model.cpu()
optimizer = optim.Adam(teacher_model.parameters(), lr=cmd_args.learning_rate)
optimizer = optim.Adam(student_model.parameters(), lr=cmd_args.learning_rate)

train_idxes = list(range(len(train_graphs)))


dataset_used = data_name
evaluator = Evaluator(name='ogbl-collab')

# print("#####################\n")
# print("student model parameters")
# print("#####################\n")
# print(student_model.parameters())

for param in student_model.parameters():
    param.data = torch.randn_like(param.data)
#Testing Teacher before KD


print("\n#####################")
print("Testing Teacher Accuracy before KD")
print("#####################\n")
teacher_model.eval()
val_loss,hits = loop_dataset(eval_graphs, teacher_model, list(range(len(eval_graphs))),evaluator)
if not hyperparameters_seal.printAUC:
    val_loss[2] = 0.0
print('\033[93maverage validation of epoch %d: loss %.5f acc %.5f auc %.5f hits@20 %.5f hits@50 %.5f hits@100 %.5f\033[0m' % (
    0, val_loss[0], val_loss[1], val_loss[2],hits[0],hits[1],hits[2]))
teacher_val_loss = val_loss[2]

print("\n#####################")
print("Testing Teacher Watermark Accuracy before KD")
print("#####################\n")
val_watermark_loss = None
val_watermark_loss,hits = loop_dataset(val_watermark_graphs, teacher_model, list(range(len(val_watermark_graphs))),evaluator)
if not hyperparameters_seal.printAUC:
    val_watermark_loss[2] = 0.0
print('\033[94maverage test of epoch %d: loss %.5f acc %.5f auc %.5f hits@20 %.5f hits@50 %.5f hits@100 %.5f\033[0m' % (
    0, val_watermark_loss[0], val_watermark_loss[1], val_watermark_loss[2],hits[0],hits[1],hits[2]))
teacher_watermark_loss = val_watermark_loss[2]

#Testing Student before KD
print("\n#####################\n")
print("Testing Student Accuracy before KD")
print("#####################\n")
student_model.eval()
val_loss,hits = loop_dataset(eval_graphs, student_model, list(range(len(eval_graphs))),evaluator)
if not hyperparameters_seal.printAUC:
    val_loss[2] = 0.0
print('\033[93maverage validation of epoch %d: loss %.5f acc %.5f auc %.5f hits@20 %.5f hits@50 %.5f hits@100 %.5f\033[0m' % (
    0, val_loss[0], val_loss[1], val_loss[2],hits[0],hits[1],hits[2]))
student_auc_before = val_loss[2]

print("\n#####################")
print("Testing Student Watermark Accuracy before KD")
print("#####################\n")
val_watermark_loss = None
val_watermark_loss,hits = loop_dataset(val_watermark_graphs, student_model, list(range(len(val_watermark_graphs))),evaluator)
if not hyperparameters_seal.printAUC:
    val_watermark_loss[2] = 0.0
print('\033[94maverage test of epoch %d: loss %.5f acc %.5f auc %.5f hits@20 %.5f hits@50 %.5f hits@100 %.5f\033[0m' % (
    0, val_watermark_loss[0], val_watermark_loss[1], val_watermark_loss[2],hits[0],hits[1],hits[2]))
student_watermark_auc_before = val_watermark_loss[2]





print("\n#####################")
print("Starting to distill Knowledge")
print("#####################\n")

for epoch in range(num_epochs):
    student_model.train()
    teacher_model.eval()
    avg_loss,hits = kd_loop_dataset(
        train_graphs, teacher_model, student_model, train_idxes, evaluator,optimizer=optimizer, bsize=batch_size
    )
    print(epoch,'/',num_epochs)
    print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f hits@20 %.5f hits@50 %.5f hits@100 %.5f\033[0m' % (
    epoch, avg_loss[0], avg_loss[1], avg_loss[2],hits[0],hits[1],hits[2]))

    torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)




#Testing Student before KD
print("\n#####################")    
print("Testing Student Accuracy After KD")
print("#####################\n")
student_model.eval()
val_loss,hits = loop_dataset(eval_graphs, student_model, list(range(len(eval_graphs))),evaluator)
if not hyperparameters_seal.printAUC:
    val_loss[2] = 0.0
print('\033[93maverage validation of epoch %d: loss %.5f acc %.5f auc %.5f hits@20 %.5f hits@50 %.5f hits@100 %.5f\033[0m' % (
    0, val_loss[0], val_loss[1], val_loss[2],hits[0],hits[1],hits[2]))
student_auc_after = val_loss[2]

print("\n#####################")
print("Testing Student Watermark Accuracy After KD")
print("#####################\n")
val_watermark_loss = None
val_watermark_loss,hits = loop_dataset(val_watermark_graphs, student_model, list(range(len(val_watermark_graphs))),evaluator)
if not hyperparameters_seal.printAUC:
    val_watermark_loss[2] = 0.0
print('\033[94maverage test of epoch %d: loss %.5f acc %.5f auc %.5f hits@20 %.5f hits@50 %.5f hits@100 %.5f\033[0m' % (
    0, val_watermark_loss[0], val_watermark_loss[1], val_watermark_loss[2],hits[0],hits[1],hits[2]))
student_watermark_auc_after = val_watermark_loss[2]

with open('SEAL_KD_results_table.txt', 'w') as file:
    # Write Test AUC results
    file.write(f"Dataset:{data_name}\n")
    file.write(f"Epochs:{num_epochs}\n")
    file.write(f"Teacher :\n")
    file.write("Test AUC\n")
    file.write(f"{teacher_val_loss}\n\n")
    file.write("Watermark AUC\n")
    file.write(f"{teacher_watermark_loss}\n\n")

    file.write(f"Student :\n")
    file.write("Test AUC\n")
    file.write("Before | After\n")
    file.write(f"{student_auc_before} | {student_auc_after}\n\n")

    # Write Watermark AUC results
    file.write("Watermark AUC\n")
    file.write("Before | After\n")
    file.write(f"{student_watermark_auc_before} | {student_watermark_auc_after}\n\n")