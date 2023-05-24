from torch.nn import Linear
import torch.nn.functional as F
from utils import *
from torch import nn
from torch.nn import Parameter
from source import GCNConv
from torch_geometric.nn import GINConv, SAGEConv
from torch.nn.utils import spectral_norm
from scipy.sparse import coo_matrix
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from homophily import edge_homophily, node_homophily, class_homophily, aggregation_homophily, node_homophily_abs

class channel_masker(nn.Module):
    def __init__(self, args):
        super(channel_masker, self).__init__()

        self.weights = nn.Parameter(torch.distributions.Uniform(
            0, 1).sample((args.num_features, 2)))

    def reset_parameters(self):
        self.weights = torch.nn.init.xavier_uniform_(self.weights)

    def forward(self):
        return self.weights

# def edge_homophily(graph, labels, ignore_negative=False):
#     src_node, targ_node = graph[0], graph[1]
#     matching = labels[src_node] == labels[targ_node]
#     # labeled_mask = (labels[src_node] >= 0) * (labels[targ_node] >= 0)
#
#     edge_hom = torch.mean(matching.float())
#     # if ignore_negative:
#     #     edge_hom = np.mean(matching[labeled_mask])
#     # else:
#     #     edge_hom = np.mean(matching)
#     return edge_hom

class MLP_discriminator(torch.nn.Module):
    def __init__(self, args):
        super(MLP_discriminator, self).__init__()
        self.args = args

        self.lin = Linear(args.hidden, 1)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, h, edge_index=None, mask_node=None):
        h = self.lin(h)

        return torch.sigmoid(h)


class MLP_encoder(torch.nn.Module):
    def __init__(self, args):
        super(MLP_encoder, self).__init__()
        self.args = args

        self.lin = Linear(args.num_features, args.hidden)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def clip_parameters(self, channel_weights):
        for i in range(self.lin.weight.data.shape[1]):
            self.lin.weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                   self.args.clip_e * channel_weights[i])

        # self.lin.weight.data[:,
        #                      channels].clamp_(-self.args.clip_e, self.args.clip_e)
        # self.lin.weight.data.clamp_(-self.args.clip_e, self.args.clip_e)

    def forward(self, x, edge_index=None, mask_node=None):
        h = self.lin(x)

        return h


class GCN_encoder_scatter(torch.nn.Module):
    def __init__(self, args):
        super(GCN_encoder_scatter, self).__init__()

        self.args = args

        self.lin = Linear(args.num_features, args.hidden, bias=False)

        self.bias = Parameter(torch.Tensor(args.hidden))

    def clip_parameters(self, channel_weights):
        for i in range(self.lin.weight.data.shape[1]):
            self.lin.weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                   self.args.clip_e * channel_weights[i])

        # self.lin.weight.data[:,
        #                      channels].clamp_(-self.args.clip_e, self.args.clip_e)
        # self.lin.weight.data.clamp_(-self.args.clip_e, self.args.clip_e)

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.fill_(0.0)

    def forward(self, x, edge_index, adj_norm_sp):
        h = self.lin(x)
        h = propagate2(h, edge_index) + self.bias

        return h


class GCN_2(nn.Module):
    def __init__(self, args):
        super(GCN_2, self).__init__()
        self.body = GCN_Body(args.num_features, args.hidden, args.dropout)
        self.fc = nn.Linear(args.hidden, args.hidden)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def reset_parameters(self):
        for m in self.modules():
            self.weights_init(m)

    def forward(self, x, edge_index, adj):
        x = self.body(x, edge_index)
        x = self.fc(x)
        return x


class GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_Body, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        return x



class GCN_encoder_spmm(torch.nn.Module):
    def __init__(self, args):
        super(GCN_encoder_spmm, self).__init__()

        self.args = args

        self.lin = Linear(args.num_features, args.hidden, bias=False)

        self.bias = Parameter(torch.Tensor(args.hidden))

    def clip_parameters(self, channel_weights):
        for i in range(self.lin.weight.data.shape[1]):
            self.lin.weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                   self.args.clip_e * channel_weights[i])

        # self.lin.weight.data[:,
        #                      channels].clamp_(-self.args.clip_e, self.args.clip_e)
        # self.lin.weight.data.clamp_(-self.args.clip_e, self.args.clip_e)

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.fill_(0.0)

    def forward(self, x, edge_index, adj_norm_sp):
        h = self.lin(x)
        h = torch.spmm(adj_norm_sp, h) + self.bias
        # h = propagate2(h, edge_index) + self.bias

        return h


class GIN_encoder(nn.Module):
    def __init__(self, args):
        super(GIN_encoder, self).__init__()

        self.args = args

        self.mlp = nn.Sequential(
            nn.Linear(args.num_features, args.hidden),
            # nn.ReLU(),
            nn.BatchNorm1d(args.hidden),
            # nn.Linear(args.hidden, args.hidden),
        )

        self.conv = GINConv(self.mlp)

    def clip_parameters(self, channel_weights):
        for i in range(self.mlp[0].weight.data.shape[1]):
            self.mlp[0].weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                      self.args.clip_e * channel_weights[i])

        # self.mlp[0].weight.data[:,
        #                      channels].clamp_(-self.args.clip_e, self.args.clip_e)
        # self.mlp[0].weight.data.clamp_(-self.args.clip_e, self.args.clip_e)

        # for p in self.conv.parameters():
        #     p.data.clamp_(-self.args.clip_e, self.args.clip_e)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, adj_norm_sp):
        h = self.conv(x, edge_index)
        return h


class SAGE_encoder(nn.Module):
    def __init__(self, args):
        super(SAGE_encoder, self).__init__()

        self.args = args

        self.conv1 = SAGEConv(args.num_features, args.hidden, normalize=True)
        self.conv1.aggr = 'mean'
        self.transition = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(args.hidden),
            nn.Dropout(p=args.dropout)
        )
        self.conv2 = SAGEConv(args.hidden, args.hidden, normalize=True)
        self.conv2.aggr = 'mean'

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def clip_parameters(self, channel_weights):
        for i in range(self.conv1.lin_l.weight.data.shape[1]):
            self.conv1.lin_l.weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                           self.args.clip_e * channel_weights[i])

        for i in range(self.conv1.lin_r.weight.data.shape[1]):
            self.conv1.lin_r.weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                           self.args.clip_e * channel_weights[i])

        # for p in self.conv1.parameters():
        #     p.data.clamp_(-self.args.clip_e, self.args.clip_e)
        # for p in self.conv2.parameters():
        #     p.data.clamp_(-self.args.clip_e, self.args.clip_e)

    def forward(self, x, edge_index, adj_norm_sp):
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        h = self.conv2(x, edge_index)
        return h


class MLP_classifier(torch.nn.Module):
    def __init__(self, args):
        super(MLP_classifier, self).__init__()
        self.args = args

        self.lin = Linear(args.hidden, args.num_classes)

    def clip_parameters(self):
        for p in self.lin.parameters():
            p.data.clamp_(-self.args.clip_c, self.args.clip_c)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, h, edge_index=None):
        h = self.lin(h)

        return h

class Graph_Editer(nn.Module):
    def __init__(self, n, a, device):
        super(Graph_Editer, self).__init__()
        self.B = nn.Parameter(torch.FloatTensor(n, n))
        self.transFeature = nn.Linear(a, a)
        # self.transEdge = nn.Linear(a, a)
        self.device = device
        self.seed = 13

    def reset_parameters(self):
        self.transFeature.reset_parameters()

    def modify_structure(self, edge_index, A2, sens, nodes_num, drop=0.5, add=0.05):
        # in_hom = edge_homophily(edge_index, sens, ignore_negative=False)
        src_node, targ_node = edge_index[0], edge_index[1]
        matching = sens[src_node] == sens[targ_node]
        # 去掉异配边
        yipei = torch.where(matching == False)[0]
        drop_index = torch.LongTensor(random.sample(range(yipei.shape[0]), int(yipei.shape[0] * drop))).to(edge_index.device)
        yipei_drop = torch.index_select(yipei, 0, drop_index)
        keep_indices = torch.ones(src_node.shape, dtype=torch.bool)
        keep_indices[yipei_drop] = False
        n_src_node = src_node[keep_indices]
        n_targ_node = targ_node[keep_indices]

        src_node2, targ_node2 = A2.indices()[0], A2.indices()[1]
        matching2 = sens[src_node2] == sens[targ_node2]
        matching3 = src_node2 == targ_node2
        tongpei = torch.where(torch.logical_and(matching2 == True, matching3 == False) == True)[0]
        add_index = torch.LongTensor(random.sample(range(tongpei.shape[0]), int(yipei_drop.shape[0]))).to(edge_index.device)
        tongpei_add = torch.index_select(tongpei, 0, add_index)
        keep_indices = torch.zeros(src_node2.shape, dtype=torch.bool)
        keep_indices[tongpei_add] = True
        a_src_node = src_node2[keep_indices]
        a_targ_node = targ_node2[keep_indices]

        m_src_node = torch.cat((a_src_node, n_src_node))
        m_targ_node = torch.cat((a_targ_node, n_targ_node))

        n_edge_index = torch.cat((m_src_node.unsqueeze(0), m_targ_node.unsqueeze(0)), dim=0)
        # edge_index = remove_duplicates(edge_index)
        eweight = torch.ones(n_edge_index.shape[1]).to(edge_index.device)
        n_adj = torch.sparse_coo_tensor(n_edge_index, eweight, [nodes_num, nodes_num])
        n_adj_dense = n_adj.to_dense()
        sparse_n_adj = torch.sparse_coo_tensor(n_adj_dense.nonzero().T, n_adj_dense[n_adj_dense != 0],
                                               n_adj_dense.size())
        n_edge_index = sparse_n_adj.coalesce().indices()
        return n_edge_index

    def modify_structure1(self, edge_index, adj, A2, sens, nodes_num, drop=0.8, add=0.3):
        random.seed(self.seed)
        src_node, targ_node = edge_index[0], edge_index[1]
        matching = sens[src_node] == sens[targ_node]
        adj = adj.to_dense()
        # in_hom = torch.mean(matching.float())
        # edge_hom = edge_homophily(adj, sens)
        # node_hom = node_homophily(adj, sens)
        # class_hom = class_homophily(adj, sens)
        # #agg_hom = aggregation_homophily(features, adj, sens)
        # print("=======drop:{}========".format(drop))
        # print("in_hom:{}".format(in_hom))
        # print("edge_hom:{}".format(edge_hom))
        # print("node_hom:{}".format(node_hom))
        # print("class_hom:{}".format(class_hom))
        # print("agg_hom:{}".format(agg_hom))
        # 去掉异配边
        yipei = torch.where(matching == False)[0]
        drop_index = torch.LongTensor(random.sample(range(yipei.shape[0]), int(yipei.shape[0] * drop))).to(edge_index.device)
        yipei_drop = torch.index_select(yipei, 0, drop_index)
        keep_indices = torch.ones(src_node.shape, dtype=torch.bool)
        keep_indices[yipei_drop] = False
        n_src_node = src_node[keep_indices]
        n_targ_node = targ_node[keep_indices]
        # 加同配
        src_node2, targ_node2 = A2.indices()[0], A2.indices()[1]
        matching2 = sens[src_node2] == sens[targ_node2]
        matching3 = src_node2 == targ_node2
        tongpei = torch.where(torch.logical_and(matching2 == True, matching3 == False) == True)[0]
        add_index = torch.LongTensor(random.sample(range(tongpei.shape[0]), int(yipei.shape[0] * drop))).to(edge_index.device)
        tongpei_add = torch.index_select(tongpei, 0, add_index)
        keep_indices = torch.zeros(src_node2.shape, dtype=torch.bool)
        keep_indices[tongpei_add] = True
        a_src_node = src_node2[keep_indices]
        a_targ_node = targ_node2[keep_indices]

        m_src_node = torch.cat((a_src_node, n_src_node))
        m_targ_node = torch.cat((a_targ_node, n_targ_node))

        # matching = sens[m_src_node] == sens[m_targ_node]
        # out_hom = torch.mean(matching.float())


        n_edge_index = torch.cat((m_src_node.unsqueeze(0), m_targ_node.unsqueeze(0)), dim=0)
        # edge_index = remove_duplicates(edge_index)
        eweight = torch.ones(n_edge_index.shape[1]).to(edge_index.device)
        n_adj = torch.sparse_coo_tensor(n_edge_index, eweight, [nodes_num, nodes_num])
        n_adj_dense = n_adj.to_dense()
        sparse_n_adj = torch.sparse_coo_tensor(n_adj_dense.nonzero().T, n_adj_dense[n_adj_dense != 0],
                                               n_adj_dense.size())
        n_edge_index = sparse_n_adj.coalesce().indices()
        src_node, targ_node = n_edge_index[0], n_edge_index[1]
        matching = sens[src_node] == sens[targ_node]

        # in_hom = torch.mean(matching.float())
        # edge_hom = edge_homophily(n_adj_dense, sens)
        # node_hom = node_homophily(n_adj_dense, sens)
        # node_hom_abs = node_homophily_abs(n_adj_dense, sens)
        # class_hom = class_homophily(n_adj_dense, sens)
        # # agg_hom = aggregation_homophily(features, n_adj_dense, sens)
        # print("=====after======")
        # print("in_hom:{}".format(in_hom))
        # print("edge_hom:{}".format(edge_hom))
        # print("node_hom:{}".format(node_hom))
        # print("node_hom_abs:{}".format(node_hom_abs))
        # print("class_hom:{}".format(class_hom))
        # print("agg_hom:{}".format(agg_hom))

        # print(out_hom.item())


        return n_edge_index


    def modify_structure2(self, edge_index,  adj, A2, sens, nodes_num, drop=0.6, add=0.3):
        random.seed(self.seed)
        src_node, targ_node = edge_index[0], edge_index[1]
        matching = sens[src_node] == sens[targ_node]
        adj = adj.to_dense()
        # in_hom = torch.mean(matching.float())
        # edge_hom = edge_homophily(adj, sens)
        # node_hom = node_homophily(adj, sens)
        # # node_hom_abs = node_homophily_abs(adj, sens)
        # class_hom = class_homophily(adj, sens)
        # # agg_hom = aggregation_homophily(features, adj, sens)
        # print("=======drop:{}========".format(drop))
        # print("in_hom:{}".format(in_hom))
        # print("edge_hom:{}".format(edge_hom))
        # print("node_hom:{}".format(node_hom))
        # # print("node_hom_abs:{}".format(node_hom_abs))
        # print("class_hom:{}".format(class_hom))
        # print("agg_hom:{}".format(agg_hom))
        # 去掉同配边
        yipei = torch.where(matching == True)[0]
        random.shuffle(yipei)
        drop_index = torch.LongTensor(random.sample(range(yipei.shape[0]), int(yipei.shape[0] * drop))).to(edge_index.device)
        yipei_drop = torch.index_select(yipei, 0, drop_index)
        keep_indices = torch.ones(src_node.shape, dtype=torch.bool)
        keep_indices[yipei_drop] = False
        n_src_node = src_node[keep_indices]
        n_targ_node = targ_node[keep_indices]
        # 加异配
        src_node2, targ_node2 = A2.indices()[0], A2.indices()[1]
        matching2 = sens[src_node2] != sens[targ_node2]
        matching3 = src_node2 == targ_node2
        tongpei = torch.where(torch.logical_and(matching2 == True, matching3 == False) == True)[0]
        random.shuffle(tongpei)
        add_index = torch.LongTensor(random.sample(range(tongpei.shape[0]), int(yipei.shape[0] * drop))).to(edge_index.device)
        tongpei_add = torch.index_select(tongpei, 0, add_index)
        keep_indices = torch.zeros(src_node2.shape, dtype=torch.bool)
        keep_indices[tongpei_add] = True
        a_src_node = src_node2[keep_indices]
        a_targ_node = targ_node2[keep_indices]

        m_src_node = torch.cat((a_src_node, n_src_node))
        m_targ_node = torch.cat((a_targ_node, n_targ_node))

        matching = sens[m_src_node] == sens[m_targ_node]
        out_hom = torch.mean(matching.float())
        # print(in_hom.item())
        # print(out_hom.item())

        n_edge_index = torch.cat((m_src_node.unsqueeze(0), m_targ_node.unsqueeze(0)), dim=0)
        # edge_index = remove_duplicates(edge_index)
        eweight = torch.ones(n_edge_index.shape[1]).to(edge_index.device)
        n_adj = torch.sparse_coo_tensor(n_edge_index, eweight, [nodes_num, nodes_num])
        n_adj_dense = n_adj.to_dense()
        sparse_n_adj = torch.sparse_coo_tensor(n_adj_dense.nonzero().T, n_adj_dense[n_adj_dense != 0],
                                               n_adj_dense.size())
        n_edge_index = sparse_n_adj.coalesce().indices()
        src_node, targ_node = n_edge_index[0], n_edge_index[1]
        matching = sens[src_node] == sens[targ_node]

        # in_hom = torch.mean(matching.float())
        # edge_hom = edge_homophily(n_adj_dense, sens)
        # node_hom = node_homophily(n_adj_dense, sens)
        # node_hom_abs = node_homophily_abs(n_adj_dense, sens)
        # class_hom = class_homophily(n_adj_dense, sens)
        # # agg_hom = aggregation_homophily(features, n_adj_dense, sens)
        # print("=====after======")
        # print("in_hom:{}".format(in_hom))
        # print("edge_hom:{}".format(edge_hom))
        # print("node_hom:{}".format(node_hom))
        # print("node_hom_abs:{}".format(node_hom_abs))
        # print("class_hom:{}".format(class_hom))
        # # print("agg_hom:{}".format(agg_hom))



        return n_edge_index

    def forward(self, x):
        x1 = x + 0.1 * self.transFeature(x)

        return x1



class Graph_Editer2(nn.Module):
    def __init__(self, n, a, device):
        super(Graph_Editer2, self).__init__()
        self.B = nn.Parameter(torch.FloatTensor(n, n))
        self.transFeature = nn.Linear(a, a)
        # self.transEdge = nn.Linear(a, a)
        self.device = device
        self.seed = 13

    def reset_parameters(self):
        self.transFeature.reset_parameters()

    def modify_structure1(self, edge_index, features, adj, A2, sens, nodes_num, drop=0.8, add=0.3):
        random.seed(self.seed)
        src_node, targ_node = edge_index[0], edge_index[1]
        matching = sens[src_node] == sens[targ_node]
        adj = adj.to_dense()
        in_hom = torch.mean(matching.float())
        edge_hom = edge_homophily(adj, sens)
        node_hom = node_homophily(adj, sens)
        class_hom = class_homophily(adj, sens)
        agg_hom = aggregation_homophily(features, adj, sens)
        print("=======drop:{}========".format(drop))
        print("in_hom:{}".format(in_hom))
        print("edge_hom:{}".format(edge_hom))
        print("node_hom:{}".format(node_hom))
        print("class_hom:{}".format(class_hom))
        print("agg_hom:{}".format(agg_hom))
        # 去掉异配边
        yipei = torch.where(matching == False)[0]
        drop_index = torch.LongTensor(random.sample(range(yipei.shape[0]), int(yipei.shape[0] * drop))).to(edge_index.device)
        yipei_drop = torch.index_select(yipei, 0, drop_index)
        keep_indices = torch.ones(src_node.shape, dtype=torch.bool)
        keep_indices[yipei_drop] = False
        n_src_node = src_node[keep_indices]
        n_targ_node = targ_node[keep_indices]
        # 加同配
        src_node2, targ_node2 = A2.indices()[0], A2.indices()[1]
        matching2 = sens[src_node2] == sens[targ_node2]
        matching3 = src_node2 == targ_node2
        tongpei = torch.where(torch.logical_and(matching2 == True, matching3 == False) == True)[0]
        add_index = torch.LongTensor(random.sample(range(tongpei.shape[0]), int(yipei.shape[0] * drop))).to(edge_index.device)
        tongpei_add = torch.index_select(tongpei, 0, add_index)
        keep_indices = torch.zeros(src_node2.shape, dtype=torch.bool)
        keep_indices[tongpei_add] = True
        a_src_node = src_node2[keep_indices]
        a_targ_node = targ_node2[keep_indices]

        m_src_node = torch.cat((a_src_node, n_src_node))
        m_targ_node = torch.cat((a_targ_node, n_targ_node))

        # matching = sens[m_src_node] == sens[m_targ_node]
        # out_hom = torch.mean(matching.float())


        n_edge_index = torch.cat((m_src_node.unsqueeze(0), m_targ_node.unsqueeze(0)), dim=0)
        # edge_index = remove_duplicates(edge_index)
        eweight = torch.ones(n_edge_index.shape[1]).to(edge_index.device)
        n_adj = torch.sparse_coo_tensor(n_edge_index, eweight, [nodes_num, nodes_num])
        n_adj_dense = n_adj.to_dense()
        sparse_n_adj = torch.sparse_coo_tensor(n_adj_dense.nonzero().T, n_adj_dense[n_adj_dense != 0],
                                               n_adj_dense.size())
        n_edge_index = sparse_n_adj.coalesce().indices()
        src_node, targ_node = n_edge_index[0], n_edge_index[1]
        matching = sens[src_node] == sens[targ_node]

        in_hom = torch.mean(matching.float())
        edge_hom = edge_homophily(n_adj_dense, sens)
        node_hom = node_homophily(n_adj_dense, sens)
        node_hom_abs = node_homophily_abs(n_adj_dense, sens)
        class_hom = class_homophily(n_adj_dense, sens)
        agg_hom = aggregation_homophily(features, n_adj_dense, sens)
        print("=====after======")
        print("in_hom:{}".format(in_hom))
        print("edge_hom:{}".format(edge_hom))
        print("node_hom:{}".format(node_hom))
        print("node_hom_abs:{}".format(node_hom_abs))
        print("class_hom:{}".format(class_hom))
        print("agg_hom:{}".format(agg_hom))

        # print(out_hom.item())


        return n_edge_index, in_hom, edge_hom, node_hom, node_hom_abs, class_hom, agg_hom


    def modify_structure2(self, edge_index, features, adj, A2, sens, nodes_num, drop=0.6, add=0.3):
        random.seed(self.seed)
        src_node, targ_node = edge_index[0], edge_index[1]
        matching = sens[src_node] == sens[targ_node]
        adj = adj.to_dense()
        in_hom = torch.mean(matching.float())
        edge_hom = edge_homophily(adj, sens)
        node_hom = node_homophily(adj, sens)
        # node_hom_abs = node_homophily_abs(adj, sens)
        class_hom = class_homophily(adj, sens)
        agg_hom = aggregation_homophily(features, adj, sens)
        print("=======drop:{}========".format(drop))
        print("in_hom:{}".format(in_hom))
        print("edge_hom:{}".format(edge_hom))
        print("node_hom:{}".format(node_hom))
        # print("node_hom_abs:{}".format(node_hom_abs))
        print("class_hom:{}".format(class_hom))
        print("agg_hom:{}".format(agg_hom))
        # 去掉同配边
        yipei = torch.where(matching == True)[0]
        random.shuffle(yipei)
        drop_index = torch.LongTensor(random.sample(range(yipei.shape[0]), int(yipei.shape[0] * drop))).to(edge_index.device)
        yipei_drop = torch.index_select(yipei, 0, drop_index)
        keep_indices = torch.ones(src_node.shape, dtype=torch.bool)
        keep_indices[yipei_drop] = False
        n_src_node = src_node[keep_indices]
        n_targ_node = targ_node[keep_indices]
        # 加异配
        src_node2, targ_node2 = A2.indices()[0], A2.indices()[1]
        matching2 = sens[src_node2] != sens[targ_node2]
        matching3 = src_node2 == targ_node2
        tongpei = torch.where(torch.logical_and(matching2 == True, matching3 == False) == True)[0]
        random.shuffle(tongpei)
        add_index = torch.LongTensor(random.sample(range(tongpei.shape[0]), int(yipei.shape[0] * drop))).to(edge_index.device)
        tongpei_add = torch.index_select(tongpei, 0, add_index)
        keep_indices = torch.zeros(src_node2.shape, dtype=torch.bool)
        keep_indices[tongpei_add] = True
        a_src_node = src_node2[keep_indices]
        a_targ_node = targ_node2[keep_indices]

        m_src_node = torch.cat((a_src_node, n_src_node))
        m_targ_node = torch.cat((a_targ_node, n_targ_node))

        matching = sens[m_src_node] == sens[m_targ_node]
        out_hom = torch.mean(matching.float())
        # print(in_hom.item())
        # print(out_hom.item())

        n_edge_index = torch.cat((m_src_node.unsqueeze(0), m_targ_node.unsqueeze(0)), dim=0)
        # edge_index = remove_duplicates(edge_index)
        eweight = torch.ones(n_edge_index.shape[1]).to(edge_index.device)
        n_adj = torch.sparse_coo_tensor(n_edge_index, eweight, [nodes_num, nodes_num])
        n_adj_dense = n_adj.to_dense()
        sparse_n_adj = torch.sparse_coo_tensor(n_adj_dense.nonzero().T, n_adj_dense[n_adj_dense != 0],
                                               n_adj_dense.size())
        n_edge_index = sparse_n_adj.coalesce().indices()
        src_node, targ_node = n_edge_index[0], n_edge_index[1]
        matching = sens[src_node] == sens[targ_node]

        in_hom = torch.mean(matching.float())
        edge_hom = edge_homophily(n_adj_dense, sens)
        node_hom = node_homophily(n_adj_dense, sens)
        node_hom_abs = node_homophily_abs(n_adj_dense, sens)
        class_hom = class_homophily(n_adj_dense, sens)
        agg_hom = aggregation_homophily(features, n_adj_dense, sens)
        print("=====after======")
        print("in_hom:{}".format(in_hom))
        print("edge_hom:{}".format(edge_hom))
        print("node_hom:{}".format(node_hom))
        print("node_hom_abs:{}".format(node_hom_abs))
        print("class_hom:{}".format(class_hom))
        print("agg_hom:{}".format(agg_hom))



        return n_edge_index, in_hom, edge_hom, node_hom, node_hom_abs, class_hom, agg_hom

    def forward(self, x):
        x1 = x + 0.1 * self.transFeature(x)

        return x1

