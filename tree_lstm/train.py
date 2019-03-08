import argparse
import time
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.nn.init as INIT
import torch.optim as optim
from torch.utils.data import DataLoader

import dgl
import dgl.data as data

from tree_lstm import TreeLSTM

import collections

SSTBatch = collections.namedtuple('SSTBatch', ['graph', 'mask', 'wordid', 'label'])
def batcher(device):
    def batcher_dev(batch):
        batch_trees = dgl.batch(batch)
        return SSTBatch(graph=batch_trees,
                        mask=batch_trees.ndata['mask'].to(device),
                        wordid=batch_trees.ndata['x'].to(device),
                        label=batch_trees.ndata['y'].to(device))
    return batcher_dev


def main(args):
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed(args.seed)

    cuda = args.gpu >= 0
    device = th.device('cuda:{}'.format(args.gpu)) if cuda else th.device('cpu')
    if cuda:
        th.cuda.set_device(args.gpu)

    trainset = data.SST()
    train_loader = DataLoader(dataset=trainset,
                              batch_size=args.batch_size,
                              collate_fn=batcher(device),
                              shuffle=True,
                              num_workers=0)

    model = TreeLSTM(trainset.num_vocabs,
                     args.x_size,
                     args.h_size,
                     trainset.num_classes,
                     args.dropout,
                     cell_type='childsum' if args.child_sum else 'nary',
                     pretrained_emb = trainset.pretrained_emb).to(device)
    print(model)
    params_ex_emb =[x for x in list(model.parameters()) if x.requires_grad and x.size(0)!=trainset.num_vocabs]
    params_emb = list(model.embedding.parameters())

    optimizer = optim.Adagrad([
        {'params':params_ex_emb, 'lr':args.lr, 'weight_decay':args.weight_decay},
        {'params':params_emb, 'lr':0.1*args.lr}])

    for epoch in range(args.epochs):
        model.train()
        count = 0
        t_epoch = time.time()
        for step, batch in enumerate(train_loader):
            g = batch.graph
            n = g.number_of_nodes()
            h = th.zeros((n, args.h_size)).to(device)
            c = th.zeros((n, args.h_size)).to(device)

            logits = model(batch, h, c)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp, batch.label, reduction='elementwise_mean')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += 1

            if cuda:
                th.cuda.synchronize()

        t_epoch_end = time.time()

        print('Epoch {:05d} batch {} training time {:.4f}s'.format(epoch, count, t_epoch_end - t_epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=12110)
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--child-sum', action='store_true')
    parser.add_argument('--x-size', type=int, default=300)
    parser.add_argument('--h-size', type=int, default=150)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log-every', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    args = parser.parse_args()
    print(args)
    main(args)
