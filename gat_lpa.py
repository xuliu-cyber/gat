# 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import dgl
import os
import argparse
import math
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair

epsilon = 1 - math.log(2)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--Lambda', type=float, default=1,
                    help='Please refer to Eqn. (18) in original paper')
args = parser.parse_args()

class ElementWiseLinear(nn.Module):
    def __init__(self, size, weight=True, bias=True, inplace=False):
        super().__init__()
        if weight:
            self.weight = nn.Parameter(torch.Tensor(size))
        else:
            self.weight = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(size))
        else:
            self.bias = None
        self.inplace = inplace

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.inplace:
            if self.weight is not None:
                x.mul_(self.weight)
            if self.bias is not None:
                x.add_(self.bias)
        else:
            if self.weight is not None:
                x = x * self.weight
            if self.bias is not None:
                x = x + self.bias
        return x

class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 edge_drop=0.0,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, y, get_attention=False):
        r"""

        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope(): # not override the existing data
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            
            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(-1, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)

            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute e_ij --initial attention weight

            # drop edge
            if self.training and self.edge_drop > 0:
                perm = torch.randperm(graph.number_of_edges(), device=e.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
                graph.edata["a"] = torch.zeros_like(e)
                graph.edata["a"][eids] = self.attn_drop(edge_softmax(graph, e[eids], eids=eids))
            # compute softmax
            else:
                graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']

            # LPA prediction
            graph.srcdata.update({'label': y})
            graph.update_all(fn.u_mul_e('label', 'a', 'm'),
                             fn.sum('m', 'label'))
            y_pred = graph.dstdata['label']
            '''
            # to restore the attention matrix
            node_num = len(graph.nodes())
            indices = torch.stack(list(graph.edges()), dim=0)
            values = torch.mean(graph.edata['a'], dim=1).squeeze()
            indices, values = indices.to(graph.device), values.to(graph.device)
            # use attention matrix to predict labels
            # attn_mat = torch.sparse_coo_tensor(indices, values.squeeze(),(node_num, node_num),device=graph.device)
            # y_pred = torch.sparse.mm(attn_mat, y)
            '''

            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], self._num_heads, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(1, self._num_heads, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, y_pred, graph.edata['a']
            else:
                return rst, y_pred
            

class GAT(nn.Module):
    def __init__(
        self,
        in_feats,
        n_classes,
        n_hidden,
        n_layers,
        n_heads,
        activation,
        dropout=0.0,
        input_drop=0.0,
        attn_drop=0.0,
        edge_drop=0.0,
        #use_attn_dst=True,
        #use_symmetric_norm=False,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.num_heads = n_heads

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.linear = nn.ModuleList()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            num_heads = n_heads if i < n_layers - 1 else 1
            out_channels = n_heads

            self.convs.append(
                GATConv(
                    in_hidden,
                    out_hidden,
                    num_heads=num_heads,
                    attn_drop=attn_drop,
                    edge_drop=edge_drop,
                    #use_attn_dst=use_attn_dst,
                    #use_symmetric_norm=use_symmetric_norm,
                    residual=True,
                )
            )

            if i < n_layers - 1:
                self.norms.append(nn.BatchNorm1d(out_channels * out_hidden))

        self.bias_last = ElementWiseLinear(n_classes, weight=False, bias=True, inplace=True)

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat, y):
        h = feat
        h = self.input_drop(h)
        graph = graph.to(self.device)
        h = h.to(self.device)

        for i in range(self.n_layers):
            conv, y = self.convs[i](graph, h, y)

            h = conv
            if i < self.n_layers - 1:
                h = h.flatten(1)
                h = self.norms[i](h)
                h = self.activation(h, inplace=True)
                h = self.dropout(h)

        h = h.mean(1)
        h = self.bias_last(h)

        # take log_softmax for y?
        y = y.mean(1)
        y = F.log_softmax(y, dim=1)
        return h, y

""" class GAT(nn.Module):
    def __init__(self, g, num_layers, in_dim, num_hidden, num_classes, heads, activation, feat_drop, attn_drop, negative_slope, residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.gat_layers.append(GATConv(in_dim, num_hidden, heads[0], feat_drop, attn_drop, negative_slope, False, self.activation))
        for l in range(1, num_layers):
            self.gat_layers.append(GATConv(num_hidden * heads[l-1], num_hidden, heads[l], feat_drop, attn_drop, negative_slope, residual, self.activation))
        self.gat_layers.append(GATConv(num_hidden * heads[-2], num_classes, heads[-1], feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs):
        h=inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits
 """

def preprocess(graph):
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    graph.create_formats_()
    return graph
    
def one_hot_embedding(labels, num_classes):
    # Embedding labels to one-hot form.
    y = torch.eye(num_classes) 
    return y[labels] 

# 损失函数 lce = log(eps + Lce)
def Logarithmic_Cross_Entropy(x, labels):
    y = F.cross_entropy(x, labels, reduction="none")
    y = torch.log(epsilon + y) - math.log(epsilon)
    return torch.mean(y)

def train(model, g,features, train_mask, labels, y, optimizer):
    model.train()
    optimizer.zero_grad()

    g = g.to(model.device)
    features = features.to(model.device)

    out, y_pred = model(g, features, y)
    out, y_pred = out.to(model.device), y_pred.to(model.device)
    labels = labels.to(model.device)

    # gat-lcp loss
    loss_gat = Logarithmic_Cross_Entropy(out[train_mask], labels[train_mask].squeeze())
    loss_lcp = Logarithmic_Cross_Entropy(y_pred[train_mask], labels[train_mask].squeeze())
    loss = loss_gat + args.Lambda * loss_lcp
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, g,features, train_mask, val_mask, test_mask, labels, y, evaluator):
    model.eval()
    out, y_pred = model(g, features, y)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': labels[train_mask],
        'y_pred': y_pred[train_mask],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': labels[val_mask],
        'y_pred': y_pred[val_mask],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': labels[test_mask],
        'y_pred': y_pred[test_mask],
    })['acc']

    return train_acc, valid_acc, test_acc

def load_data():
    dataset = DglNodePropPredDataset(name = 'ogbn-arxiv')
    g, labels = dataset[0]
    evaluator = Evaluator(name = 'ogbn-arxiv')
    split_idx = dataset.get_idx_split()
    train_mask,val_mask,test_mask=split_idx['train'],split_idx['valid'],split_idx['test']
    return g, labels, train_mask, val_mask, test_mask, evaluator

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    n_hidden = 250
    n_layers = 3
    n_heads = 3
    dropout = 0.75
    input_drop = 0.1
    attn_drop = 0.0
    edge_drop = 0.1
    n_runs = 3
    model_save_step = 100
    model_save_dir = './models'
    test_accs, val_accs = [], []

    for i in range(n_runs):
        g, labels, train_mask, val_mask, test_mask, evaluator = load_data()
        g = preprocess(g)
        features = g.ndata['feat']
        n_feats = features.shape[1]
        n_classes = (labels.max() + 1).item()
        g, labels, train_mask, val_mask, test_mask = map(
            lambda x: x.to(device), (g, labels, train_mask, val_mask, test_mask)
        )
    
        labels_for_lpa = one_hot_embedding(labels, labels.max().item() + 1).type(torch.FloatTensor).to(device)
        labels_for_lpa = labels_for_lpa.squeeze()
    
        # zero vectors for unlabeled nodes
        labels_for_lpa[val_mask] = labels_for_lpa[test_mask] = 0
        model = GAT(n_feats, n_classes, n_hidden, n_layers, n_heads, F.relu, dropout,input_drop,attn_drop,edge_drop)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.002, weight_decay = 0)
        best_val_acc, best_test_acc = 0, 0
        
        for epoch in range(500):
            loss = train(model, g, features, train_mask, labels, labels_for_lpa, optimizer)
            result = test(model, g, features, train_mask, val_mask, test_mask, labels, labels_for_lpa, evaluator)
            train_acc, valid_acc, test_acc = result
            
            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                best_test_acc = test_acc

            if (epoch+1) % model_save_step == 0:
                GAT_path = os.path.join(model_save_dir, '{}-GAT.ckpt'.format(epoch+1))
                torch.save(model.state_dict(), GAT_path)
                print('Saved model checkpoints into {}...'.format(model_save_dir))
            
            print(  f'Epoch: {epoch+1:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * train_acc:.2f}%, '
                    f'Valid: {100 * valid_acc:.2f}% '
                    f'Test: {100 * test_acc:.2f}%')
        
        print(f"Runned {i+1} times,"
              f'Valid_Acc: {100 * best_val_acc:.2f}%, '
              f'Test_Acc: {100 * best_test_acc:.2f}%')
        val_accs.append(best_val_acc)
        test_accs.append(best_test_acc)

    print(f"Runned {n_runs} times in total")
    print(f"Average val accuracy: {np.mean(val_accs)} ± {np.std(val_accs)}")
    print(f"Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")