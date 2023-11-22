#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : model.py
# @Time      : 2022/01/01 22:15:22
# @Author    : Zhao-Wenny

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATConv
from torch.nn import Parameter
#import sparse as sp

class MMConv(nn.Module):
    def __init__(self, in_features, out_features,  moment=4, use_center_moment=False,use_quantiles=False,use_resnet=True):
        super(MMConv, self).__init__() 
        self.moment = moment
        self.use_center_moment = use_center_moment
        self.use_quantiles = use_quantiles
        self.in_features = in_features

        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.w_att = Parameter(torch.FloatTensor(self.in_features * 2,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        self.w_att.data.uniform_(-stdv, stdv)
    def moment_calculation(self, x, adj_t, moment):
        mu = torch.spmm(adj_t, x)
        out_list = [mu]
        if moment > 1:
            if self.use_center_moment:
                sigma = torch.spmm(adj_t, (x - mu).pow(2))
            else:
                sigma = torch.spmm(adj_t, (x).pow(2))
            sigma[sigma == 0] = 1e-16
            sigma = sigma.sqrt()
            out_list.append(sigma)

            for order in range(3, moment+1):
                gamma = torch.spmm(adj_t, x.pow(order))
                mask_neg = None
                if torch.any(gamma == 0):
                    gamma[gamma == 0] = 1e-16
                if torch.any(gamma < 0):
                    mask_neg = gamma < 0
                    gamma[mask_neg] *= -1
                gamma = gamma.pow(1/order)
                if mask_neg != None:
                    gamma[mask_neg] *= -1
                out_list.append(gamma)
        if self.use_quantiles:
            q1_list = []
            q2_list = []
            q3_list = []
                
            for v in range(x.size(0)):
                neighbors = adj_t[v]._indices().to_dense()[0]
                neighbor_values = x[neighbors]
                q1 = torch.quantile(neighbor_values, 0.25, dim=0)
                q2 = torch.quantile(neighbor_values, 0.5, dim=0)
                q3 = torch.quantile(neighbor_values, 0.75, dim=0)
                    
                q1_list.append(q1)
                q2_list.append(q2)
                q3_list.append(q3)
                    
            q1_list = torch.stack(q1_list)
            q2_list = torch.stack(q2_list)
            q3_list = torch.stack(q3_list)
        
            out_list.append(q1_list)
            out_list.append(q2_list)
            out_list.append(q3_list)
        return out_list
    def attention_layer(self, moments, q):
            k_list = []
            # if self.use_norm:
            #     h_self = self.norm(h_self) # ln
            if self.use_quantiles:
                q = q.repeat(self.moment+3, 1)
                k_list = moments
            else:
                q = q.repeat(self.moment, 1) # N * m, D
                # output for each moment of 1st-neighbors
                k_list = moments
            attn_input = torch.cat([torch.cat(k_list, dim=0), q], dim=1)
            attn_input = F.dropout(attn_input, 0.5, training=self.training)
            e = F.elu(torch.mm(attn_input, self.w_att)) # N*m, D
            attention = F.softmax(e.view(len(k_list), -1, self.out_features).transpose(0, 1), dim=1) # N, m, D
            out = torch.stack(k_list, dim=1).mul(attention).sum(1) # N, D
            return out
    def forward(self, input, adj , h0 , lamda, alpha, l, beta=0.9):
        theta = math.log(lamda/l+1)
        h_agg = torch.spmm(adj, input)
        h_agg = (1-alpha)*h_agg+alpha*h0
        h_i = torch.mm(h_agg, self.weight)
        h_i = theta*h_i+(1-theta)*h_agg
        # h_moment = self.attention_layer(self.moment_calculation(input, adj, self.moment), h_i)
        h_moment = self.attention_layer(self.moment_calculation(h0, adj, self.moment), h_i)
        output = (1 - beta) * h_i + beta * h_moment
        return output

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)

        return (beta * z).sum(1), beta


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, mask=None):
        u = torch.matmul(q, k.transpose(-2, -1))
        u = u / self.scale

        if mask is not None:
            u = u.masked_fill(mask, -np.inf)

        attn = self.softmax(u)
        output = torch.matmul(attn, v)

        return attn, output


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, n_head, d_k_, d_v_, d_k, d_v, d_o):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.fc_q = nn.Linear(d_k_, n_head * d_k)
        self.fc_k = nn.Linear(d_k_, n_head * d_k)
        self.fc_v = nn.Linear(d_v_, n_head * d_v)

        self.attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))

        self.fc_o = nn.Linear(n_head * d_v, d_o)

    def forward(self, q, k, v, mask=None):

        n_head, d_q, d_k, d_v = self.n_head, self.d_k, self.d_k, self.d_v

        n_q, d_q_ = q.size()
        n_k, d_k_ = k.size()
        n_v, d_v_ = v.size()

        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)
        q = q.view(n_q, n_head, d_q).permute(
            1, 0, 2).contiguous().view(-1, n_q, d_q)
        k = k.view(n_k, n_head, d_k).permute(
            1, 0, 2).contiguous().view(-1, n_k, d_k)
        v = v.view(n_v, n_head, d_v).permute(
            1, 0, 2).contiguous().view(-1, n_v, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        attn, output = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, n_q, d_v).permute(
            0, 1, 2).contiguous().view(n_q, -1)
        output = self.fc_o(output)

        return attn, output


class SelfAttention(nn.Module):
    """ Self-Attention """

    def __init__(self, n_head, d_k, d_v, d_x, d_o):
        super().__init__()
        self.wq = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wk = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wv = nn.Parameter(torch.Tensor(d_x, d_v))

        self.mha = MultiHeadAttention(
            n_head=n_head, d_k_=d_k, d_v_=d_v, d_k=d_k, d_v=d_v, d_o=d_o)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / np.power(param.size(-1), 0.5)
            param.data.uniform_(-stdv, stdv)

    def forward(self, x, mask=None):
        q = torch.matmul(x, self.wq)
        k = torch.matmul(x, self.wk)
        v = torch.matmul(x, self.wv)

        attn, output = self.mha(q, k, v, mask=mask)

        return output

# 网络融合模块


class MMGNN(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, n_output, dropout, lamda, alpha, use_center_moment=False,use_quantiles=False,use_resnet=True, moment=4):
        super(MMGNN, self).__init__()
        self.convs = nn.ModuleList()
        self.use_quantiles = use_quantiles
        self.use_resnet = use_resnet
        for _ in range(nlayers):
            self.convs.append(MMConv(nhidden, nhidden, use_center_moment=use_center_moment, moment=moment,use_quantiles=use_quantiles,use_resnet=use_resnet))
        self.fcs = nn.ModuleList()
        if self.use_resnet:
            self.fcs.append(nn.Linear(nfeat, nhidden))
            self.fcs.append(nn.Linear(nhidden*nlayers, n_output))
        else:
            self.fcs.append(nn.Linear(nfeat, nhidden))
            self.fcs.append(nn.Linear(nhidden, n_output))

        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
    
    def reset_parameters(self):
        for fc in self.fcs:
            fc.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        h = self.act_fn(self.fcs[0](x))
        _layers.append(h)
        if self.use_resnet:
            res_list = []
            for ind, conv in enumerate(self.convs):
                # if ind // 2 == 0:
                #     h_input = F.dropout(h, self.dropout, training=self.training)
                #     h_output = self.act_fn(conv(h_input,adj,_layers[0],self.lamda,self.alpha, ind+1))
                #     h = h_input + h_output
                # else:
                #     h = F.dropout(h, self.dropout, training=self.training)
                #     h = self.act_fn(conv(h,adj,_layers[0],self.lamda,self.alpha, ind+1))
                h = F.dropout(h, self.dropout, training=self.training)

                h = self.act_fn(conv(h,adj,_layers[0],self.lamda,self.alpha, ind+1))
                res_list.append(h)
            residual = torch.cat(res_list, dim=1)
            residual = F.dropout(residual, self.dropout, training=self.training)
            residual = self.fcs[-1](residual)
            return F.log_softmax(residual, dim=1)
        else:
            for ind, conv in enumerate(self.convs):
                h = F.dropout(h, self.dropout, training=self.training)
                h = self.act_fn(conv(h,adj,_layers[0],self.lamda,self.alpha, ind+1))
            
            h = F.dropout(h, self.dropout, training=self.training)
            h = self.fcs[-1](h)
            return F.log_softmax(h, dim=1)
        

    def get_emb(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        h = self.act_fn(self.fcs[0](x))
        _layers.append(h)
        for ind, conv in enumerate(self.convs):
            h = F.dropout(h, self.dropout, training=self.training)
            h = self.act_fn(conv(h,adj,_layers[0],self.lamda,self.alpha, ind+1))
        return h




class Gat_En(nn.Module):
    def __init__(self, nfeat, hidden_size, out, dropout, lamda, alpha,nlayers):
        super(Gat_En, self).__init__()
        self.MMGNN = MMGNN(nfeat,nlayers, hidden_size, out, dropout,lamda,alpha)
        #nfeat, nlayers,nhidden, n_output, dropout, lamda, alpha
        self.dropout = dropout

    def forward(self, x,edge_index):
        # x, edge_index = data.x, data.edge_index
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gat1(x, edge_index)
        # x = F.elu(self.gat1(x, edge_index))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.gat2(x, edge_index))
        #x = F.dropout(x, self.dropout, training=self.training)
        x = self.MMGNN(x, edge_index)

        return x


class MM_GNN(nn.Module):
    def __init__(self, nfeat, hidden_size1, hidden_size2, dropout, lamda, alpha,nlayers):
        super(MM_GNN, self).__init__()
        # Multi-dimensional GAT
        self.view_GNN = Gat_En(nfeat, hidden_size1, hidden_size2, dropout,lamda, alpha,nlayers)
        
        # Joint Learning Module
        self.self_attn = SelfAttention(
            n_head=1, d_k=64, d_v=32, d_x=hidden_size2, d_o=hidden_size2)
        self.attn = Attention(hidden_size2)
        
        # MLP Classifier
        self.MLP = nn.Linear(hidden_size2, 1)

        self.dropout = dropout

    def forward(self, x, edge_idex):
        #x是五张图所有节点初始特征构成的特征集合 edgeidex五张图所有index（稀疏形式）构成的集合
        embs = []
        for i in range(len(edge_idex)):
            emb = self.view_GNN(x,edge_idex[i])
            embs.append(emb)
        output = self.MLP(emb)

        return output
