#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : main.py
# @Time      : 2022/01/01 22:20:17
# @Author    : Zhao-Wenny

import argparse
import os


import numpy as np
import pandas as pd

import pandas as pd5
import sklearn.metrics as metrics
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
import scipy.sparse as sp
# from torch.utils.tensorboard.writer import SummaryWriter

from mmgnn import MM_GNN
from utils import *
from sklearn.model_selection import StratifiedKFold

cuda = torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train MM_GNN with cross-validation and save model to file')
    parser.add_argument('-e', '--epochs', help='maximum number of epochs (default: 1000)',
                        dest='epochs',
                        default=1000, #原论文为1000
                        type=int
                        )
    parser.add_argument('-p', '--patience', help='patience (default: 20)',
                        dest='patience',
                        default=100, # 20
                        type=int
                        )
    parser.add_argument('-dp', '--dropout', help='the dropout rate (default: 0.25)',
                        dest='dp',
                        default=0.6,
                        type=float
                        )
    parser.add_argument('-lr', '--learningrate', help='the learning rate (default: 0.001)',
                        dest='lr',
                        default=0.005, #0.05
                        type=float
                        )
    parser.add_argument('-wd', '--weightdecay', help='the weight decay (default: 0.0005)',
                        dest='wd',
                        default=0.0005,
                        type=float
                        )
    parser.add_argument('-hs1', '--hiddensize1', help='the hidden size of first convolution layer (default: 300)',
                        dest='hs1',
                        default=300,
                        type=int
                        )
    parser.add_argument('-hs2', '--hiddensize2', help='the hidden size of second convolution layer (default: 100)',
                        dest='hs2',
                        default=100,
                        type=int
                        )
    parser.add_argument('-seed', '--seed', help='the random seed (default: 42)',
                        dest='seed',
                        default=42,
                        type=int
                        )
    args = parser.parse_args()
    return args


def main(args):

    print('Network INFO')

    # ppi_path = os.path.join(graph_path, args['ppi'] + '_ppi.tsv')
    # go_path = os.path.join(
    #     graph_path, args['ppi'] + '_' + str(args['thr_go']) + '_go.tsv')
    # exp_path = os.path.join(
    #     graph_path, args['ppi'] + '_' + str(args['thr_exp']) + '_exp.tsv')
    # seq_path = os.path.join(
    #     graph_path, args['ppi'] + '_' + str(args['thr_seq']) + '_seq.tsv')
    # path_path = os.path.join(
    #     graph_path, args['ppi'] + '_' + str(args['thr_path']) + '_path.tsv')
    #
    # omic_path = os.path.join(graph_path, args['ppi'] + '_omics.tsv')
    #
    # if os.path.exists(ppi_path) & os.path.exists(go_path) & os.path.exists(exp_path) & os.path.exists(seq_path) & os.path.exists(path_path) & os.path.exists(omic_path):
    #     print('The five gene similarity profiles and omic feature already exist!')
    #     ppi_network = pd.read_csv(ppi_path, sep='\t', index_col=0)
    #     omicsfeature = pd.read_csv(omic_path, sep='\t', index_col=0)
    #     final_gene_node = list(omicsfeature.index)
    #
    # else:
    #     print("you chose else")
    #     omicsfeature, final_gene_node = modig_input.get_node_omicfeature()
    #     ppi_network, go_network, exp_network, seq_network, path_network = modig_input.generate_graph(
    #         args['thr_go'], args['thr_exp'], args['thr_seq'], args['thr_path'])

    print("==========================================================")
    print('Network INFO')
    # name_of_network = ['PPI', 'GO', 'EXP', 'SEQ', 'PATH']
    # network_path = ["./data_raw/PPI_for_pyG.txt", "./data_raw/Pathway_for_pyG.txt", "./data_raw/Complexes_for_pyG.txt", "./data_raw/Kinase_for_pyG.txt", "./data_raw/Metabolic_for_pyG.txt", "./data_raw/Regulatory_for_pyG.txt"]
    # graphlist = modig_input.generate_my_PyG_graph(network_path)
    # n_fdim = graphlist[0].x.shape[1]  # n_gene = featured_gsn.x.shape[0]
    # graphlist_adj = [graph.cuda() for graph in graphlist]
    name = 'bioplex'

    network_path = ['data/'+name+'_ppi.pt']
    graphlist = []
    
    for i in network_path:
        loaded_data = torch.load(i)
        graphlist.append(loaded_data)
    
    
 
    features = torch.load("data/"+name+"_feature_with_rw10.pt")
    features = features.cuda()
    
    n_fdim = features.shape[1]       
    graphs = [graph.coalesce().cuda() for graph in graphlist]
    
    labels = pd.read_csv("data/"+name+"_label.csv")
    idx_list = np.array(labels["Hugosymbol"])
    labels_list = np.array(labels["Label"])


    #上面那一串可以换成自己的数据读取组织方式，获得变量features,graphs
    print("==========================================================")

    def train(mask, label):
        model.train()
        optimizer.zero_grad()
        output = model(features,graphs)#把五张图节点的特征组织到features里面，图也是
        loss = F.binary_cross_entropy_with_logits(
            output[mask], label, pos_weight=torch.Tensor([1.74]).cuda())


        acc = metrics.accuracy_score(label.cpu(), np.round(
            torch.sigmoid(output[mask]).cpu().detach().numpy()))
        loss.backward()
        optimizer.step()
        print('train loss: ', loss.detach().cpu().item())
        del output
        return loss.item(), acc

    @torch.no_grad()
    def test(mask, label):
        model.eval()
        output = model(features,graphs)
        loss = F.binary_cross_entropy_with_logits(
            output[mask], label, pos_weight=torch.Tensor([1.74]).cuda())

        acc = metrics.accuracy_score(label.cpu(), np.round(
            torch.sigmoid(output[mask]).cpu().detach().numpy()))
        pred = torch.sigmoid(output[mask]).cpu().detach().numpy()
        auroc = metrics.roc_auc_score(label.to('cpu'), pred)
        pr, rec, _ = metrics.precision_recall_curve(label.to('cpu'), pred)
        aupr = metrics.auc(rec, pr)
        print('test loss:',loss.detach().cpu().item(),'acc:',acc,\
              'auroc:',auroc)
        return pred, loss.item()

    lamda=0.5
    alpha=0.9
    nlayers=1
    i = 1
    pred_all = []
    label_all = []
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, test_index in cv.split(idx_list, labels_list):

        print(f"第{i}折")
        X_train, X_test = torch.LongTensor(idx_list[train_index]).cuda(), torch.LongTensor(idx_list[test_index]).cuda()
        y_train, y_test = torch.FloatTensor(labels_list[train_index]).reshape(-1, 1).cuda(), torch.FloatTensor(labels_list[test_index]).reshape(-1, 1).cuda()
        
        model = MM_GNN(nfeat=n_fdim, hidden_size1=args['hs1'], hidden_size2=args['hs2'], dropout=args['dp'],lamda=lamda, alpha=alpha,nlayers=nlayers)
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
        
        early_stopping = EarlyStopping(patience=args['patience'], verbose=True)

        for epoch in range(1, args['epochs']+1):
            _, _ = train(X_train, y_train)
            _, loss_val = test(X_test, y_test)

            early_stopping(loss_val, model)
            if early_stopping.early_stop:
                print(f"Early stopping at the epoch {epoch}")
                break
            
            torch.cuda.empty_cache()
        
        y_pred, _ = test(X_test, y_test)

        
        # 将预测结果和真实标签添加到列表中
        pred_all.extend(y_pred)
        label_all.extend(y_test)

        i = i + 1
    pred_all = [float(arr[0]) for arr in pred_all]
    label_all = [float(arr[0]) for arr in label_all]
    auroc = metrics.roc_auc_score(label_all, pred_all)
    precision, recall, thresholds = precision_recall_curve(label_all, pred_all)
    auprc = auc(recall, precision)

    with open(name+'_pred.txt', 'w') as file:
        for item in pred_all:
            file.write(str(item) + '\n')

    with open(name+'_label.txt', 'w') as file:
        for item in label_all:
            file.write(str(item) + '\n')

    print(f"AUROC Score: {auroc}")
    print(f"AUPRC Score: {auprc}")

    # print('Mean AUC', AUC.mean())
    # print('Var AUC', AUC.var())
    # print('Mean AUPR', AUPR.mean())
    # print('Var AUPR', AUPR.var())
    # print('Mean ACC', ACC.mean())
    # print('Var ACC', ACC.var())

    # torch.save(pred_all, os.path.join(file_save_path, 'pred_all.pkl'))
    # torch.save(label_all, os.path.join(file_save_path, 'label_all.pkl'))

    # # Use all label to train a final model
    # all_mask = torch.LongTensor(idx_list)
    # all_label = torch.FloatTensor(label_list).reshape(-1, 1)
  
    # model = MODIG(nfeat=n_fdim, hidden_size1=args['hs1'],
    #               hidden_size2=args['hs2'], dropout=args['dp'],lamda=lamda, alpha=alpha,nlayers=nlayers)
    # model.cuda()
    # optimizer = optim.Adam(
    #     model.parameters(), lr=args['lr'], weight_decay=args['wd'])

    # for epoch in range(1, args['epochs']+1):
    #     print(epoch)
    #     _, _ = train(all_mask.cuda(), all_label.cuda())
    #     torch.cuda.empty_cache()

    # output = model(features,graphs)

    # pred = torch.sigmoid(output).cpu().detach().numpy()
    # pred2 = torch.sigmoid(output[~all_mask]).cpu().detach().numpy()
    # torch.save(pred, os.path.join(file_save_path, args['ppi'] + '_pred.pkl'))
    # torch.save(all_label, os.path.join(
    #     file_save_path, args['ppi'] + '_label.pkl'))
    # torch.save(pred2, os.path.join(file_save_path, args['ppi'] + '_pred2.pkl'))

    # # pd.Series(final_gene_node).to_csv(os.path.join(file_save_path,
    # #                                                'final_gene_node.csv'), index=False, header=False)

    # plot_average_PR_curve(pred_all, label_all, file_save_path)
    # plot_average_ROC_curve(pred_all, label_all, file_save_path)


if __name__ == '__main__':

    args = parse_args()
    args_dic = vars(args)
    print('args_dict', args_dic)

    main(args_dic)
    print('The Training is finished!')
