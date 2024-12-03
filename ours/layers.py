import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn
import numpy  as np

class Attention(nn.Module):

    def __init__(self , h , d_model):
        super().__init__()

        assert d_model % h == 0
        self.reduced_dim = 128
        self.d_model = d_model
        self.h = h
        self.dk = d_model // h
        self.nemax = 20

        self.WQ = nn.Linear(self.dk , self.dk)
        self.WK = nn.Linear(self.dk , self.dk)
        self.WV = nn.Linear(self.dk , self.dk)

        self.relative_pos_emb = nn.Parameter( torch.zeros(2 * self.nemax, 2 * self.nemax, d_model) )
        #self.reset_params()

    def forward(self , R , edge, R_mas):
        '''
            R: (bs , ne , ne , d)
            R_mas: (bs , ne , ne , 1)
        '''

        h , dk = self.h , self.dk
        bs, ne, d = R.size()

        #assert d == self.d_model or d == self.reduced_dim
        edge = torch.cat((edge, R), dim=1)
        tmpe = edge.shape[1]
        R = R.view(bs,ne,h,dk).permute(0,2,1,3).contiguous() #(bs , h , ne  , dk)

        edge = edge.view(bs,tmpe,h,dk).permute(0,2,1,3).contiguous()
        Q , K , V = self.WQ(R) , self.WK(edge) , self.WV(edge)

        Q = Q.view(bs,h,ne,dk)
        K = K.view(bs,h,tmpe,dk)
        V = V.view(bs,h,tmpe,dk)


        alpha = torch.matmul(Q , K.transpose(-1,-2))
        # print("Q", Q.shape)
        # print("K", K.shape)
        # print("alpha",alpha.shape)
        _,x1,x2 = R_mas.size()
        ###[bs, h, relations, relations+edgenode]
        att_mas = R_mas.view(1,1,x1,x2)
        alpha = alpha - (1 - att_mas) * 100000
        alpha = torch.softmax(alpha / (dk ** 0.5) , dim = -1)

        R_Z = torch.matmul(alpha , V).view(bs,h,ne,dk)

        #print("R_Z", R_Z.shape)
        R_Z = R_Z.permute(0,2,1,3).contiguous().view(bs,ne,h*dk)
        #print("R_Z", R_Z.shape)
        return R_Z

class FFN(nn.Module):
    def __init__(self , d_model , hidden_size = 1024):
        super().__init__()

        self.ln1 = nn.Linear(d_model , hidden_size)
        self.ln2 = nn.Linear(hidden_size , d_model)

        #self.reset_params()

    def reset_params(self):
        nn.init.xavier_normal_(self.ln1.weight.data)
        nn.init.xavier_normal_(self.ln2.weight.data)

        nn.init.constant_(self.ln1.bias.data , 0)
        nn.init.constant_(self.ln2.bias.data , 0)

    def forward(self , x , x_mas):
        x = F.relu(self.ln1(x))
        x = self.ln2(x)

        return x

class Encoder_Layer(nn.Module):
    def __init__(self , h , d_model , hidden_size , dropout = 0.0):
        super().__init__()

        assert d_model % h == 0

        self.d_model = d_model
        self.hidden_size = hidden_size

        self.att = Attention(h , d_model)
        self.lnorm_1 = nn.LayerNorm(d_model)
        self.drop_1 = nn.Dropout(dropout)

        self.ffn = FFN(d_model , hidden_size)
        self.lnorm_2 = nn.LayerNorm(d_model)
        self.drop_2 = nn.Dropout(dropout)


    def forward(self , R , edge_output, R_mas):
        '''
            R: (bs , ne , ne , d)
            R_mas: (bs , ne , ne , 1)
        '''

        #-----attention-----

        R_Z = self.att(R ,edge_output, R_mas)
        R = self.lnorm_1(self.drop_1(R_Z) + R)


        #-----FFN-----

        R_Z = self.ffn(R , R_mas)
        R = self.lnorm_2(self.drop_2(R_Z) + R)

        return R

class Encoder(nn.Module):
    def __init__(self , h = 4 , d_model = 768 , hidden_size = 1024 , num_layers = 6 , dropout = 0.0, device = 0):
        super().__init__()

        self.nemax = 1000
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            Encoder_Layer(h , d_model , hidden_size , dropout = dropout)
            for _ in range(num_layers)
        ])

        self.row_emb = nn.Parameter( torch.zeros(self.nemax , device = device) )
        self.col_emb = nn.Parameter( torch.zeros(self.nemax , device = device) )
        self.reset_params()

    def reset_params(self):
        nn.init.normal_(self.row_emb.data , 0 , 1e-4)
        nn.init.normal_(self.col_emb.data , 0 , 1e-4)

    def forward(self , R, edge_node, R_mas):
        '''
            R: (bs , ne , ne , d)
            sent_enc: (bs , n , d)

        '''
        #pdb.set_trace()
        # print("R", R.shape)
        # print("edge_node",edge_node.shape)
        # print("R_mas",R_mas.shape)
        ne,h = R.size()
        _,mas = R_mas.size()
        R = R.unsqueeze(0)

        R_mas = R_mas.unsqueeze(0).float()
        #print("R", R.shape)
        tmpe = edge_node.shape[0]
        edge_node = edge_node.unsqueeze(0).expand(1,tmpe,h)
        #print("edge_node",edge_node.shape)
        #R_mas = R_mas.view(1, ne, ne, 1).float()
        R = R + self.row_emb[:ne].view(1, ne, 1)

        edge_node = edge_node + self.row_emb[:tmpe].view(1, tmpe, 1)

        for layer in self.layers:
            R = layer(R , edge_node, R_mas)

        return R.squeeze(0)

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj, r):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(e * r)
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return e

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class CGGCConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, r,bias=None):
        super(CGGCConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
               
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# GRU单元的类
class GRUCell(nn.Module):
    def __init__(self, x_dim, h_dim):
        super(GRUCell, self).__init__()
        self.r = nn.Linear(x_dim + h_dim, h_dim, True)  # 记忆消除门的初始化：输入维度+隐向量的维度，输出隐向量维度
        self.z = nn.Linear(x_dim + h_dim, h_dim, True)  # 隐状态权重门的初始化：输入维度+隐向量的维度，输出隐向量维度

        self.c = nn.Linear(x_dim, h_dim, True)  # 把input变成隐状态
        self.u = nn.Linear(h_dim, h_dim, True)  # 当前隐状态

    def forward(self, x, h):
        rz_input = torch.cat((x, h), -1)
        #print("rz_input",rz_input.shape)
        r = torch.sigmoid(self.r(rz_input))
        z = torch.sigmoid(self.z(rz_input))

        u = torch.tanh(self.c(x) + r * self.u(h))

        new_h = z * h + (1 - z) * u
        return new_h


class SGRU(nn.Module):
    def __init__(self, e_emb, eh_dim, g_dim):
    #def __init__(self, e_emb, eh_dim):
        super(SGRU, self).__init__()

        self.e_gru = GRUCell(e_emb+eh_dim+g_dim, eh_dim)  # 实体编码、上图层句消息、标签、上图层全局隐向量
        #self.e_gru = GRUCell(e_emb + eh_dim, eh_dim)
        self.g_gru = GRUCell(eh_dim, g_dim)  # 上图层句消息、上图层实体消息

        self.e_nog_gru = GRUCell(e_emb+eh_dim, eh_dim)
    # s_h, e_h, g_h = self.slstm((s_input, e_input), (s_h, e_h), g_h, (smask, wmask))
    def forward(self, ei, eh, g=None):
    #def forward(self, ei, eh):
        '''
        :param it: B T 2H
        :param h: B T H
        :param g: B H
        :return:
        '''
        # update entity node
        new_g  = None
        if g is not None:
            g_expand_e = g.unsqueeze(1).expand(eh.size(0), eh.size(1), g.size(-1))
            x = torch.cat((ei, g_expand_e), -1)
            new_eh = self.e_gru(x, eh)
            eh_mean = torch.mean(new_eh,dim=1)

            new_g = self.g_gru(eh_mean,g)
            #new_eh = self.e_gru(ei, eh)
        else:
            new_eh = self.e_nog_gru(ei, eh)
        return new_eh, new_g
        #return new_eh