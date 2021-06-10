# 50%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import dgl

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
        #edge_drop=0.0,
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
                    #edge_drop=edge_drop,
                    #use_attn_dst=use_attn_dst,
                    #use_symmetric_norm=use_symmetric_norm,
                    residual=True,
                )
            )

            if i < n_layers - 1:
                self.norms.append(nn.BatchNorm1d(out_channels * out_hidden))

        #self.bias_last = ElementWiseLinear(n_classes, weight=False, bias=True, inplace=True)

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, graph, feat):
        h = feat
        h = self.input_drop(h)
        graph = graph.to(self.device)
        h = h.to(self.device)

        for i in range(self.n_layers):
            conv = self.convs[i](graph, h)

            h = conv

            if i < self.n_layers - 1:
                h = h.flatten(1)
                h = self.norms[i](h)
                h = self.activation(h, inplace=True)
                h = self.dropout(h)

        h = h.mean(1)

        return h

def train(model, g,features, train_mask, labels, optimizer):
    model.train()
    optimizer.zero_grad()
    g = g.to(model.device)
    features = features.to(model.device)
    out = model(g,features).to(model.device)
    loss = torch.nn.CrossEntropyLoss()(out[train_mask], labels[train_mask].squeeze())
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, g,features, train_mask, val_mask, test_mask, labels, evaluator):
    model.eval()
    out = model(g,features)
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

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    n_hidden = 250
    n_layers = 3
    n_heads = 3
    dropout = 0.75
    input_drop = 0.1
    attn_drop = 0.0
    edge_drop = 0.1
    reuse_num = 1
    n_runs = 3
    mask_rate = 0.5
    test_accs, val_accs = [], []
    
    for run in range(n_runs):
        g, labels, train_mask, val_mask, test_mask, evaluator = load_data()
        g = preprocess(g)
        features = g.ndata['feat']
        n_feats = features.shape[1]
        n_classes = (labels.max() + 1).item()
        g, labels, features, train_mask, val_mask, test_mask = map(
            lambda x: x.to(device), (g, labels, features, train_mask, val_mask, test_mask)
        )
    
        model = GAT(n_feats, n_classes, n_hidden, n_layers, n_heads, F.relu, dropout, input_drop, attn_drop)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.002, weight_decay = 5e-4)
        
        best_val_acc, best_test_acc = 0, 0

        for epoch in range(500):
            loss = train(model, g, features, train_mask, labels, optimizer)
            result = test(model, g, features, train_mask, val_mask, test_mask, labels, evaluator)
            train_acc, valid_acc, test_acc = result
            print(  f'Epoch: {epoch+1:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * train_acc:.2f}%, '
                    f'Valid: {100 * valid_acc:.2f}% '
                    f'Test: {100 * test_acc:.2f}%')
                    
            if valid_acc > best_val_acc:
                    best_val_acc = valid_acc
                    best_test_acc = test_acc
            
        print(f"Runned {run+1} times,"
              f'Valid_Acc: {100 * best_val_acc:.2f}%, '
              f'Test_Acc: {100 * best_test_acc:.2f}%')
        val_accs.append(best_val_acc)
        test_accs.append(best_test_acc)
    
    print(f"Runned {n_runs} times in total")
    print(f"Average val accuracy: {np.mean(val_accs)} ± {np.std(val_accs)}")
    print(f"Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")
