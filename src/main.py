'''
Top-K Diversity Regularizer

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

'''

import time
import math
import click
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from model import BPR
from utils import *

# Slice the given list into chunks of size n.
def list_chunk(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

@click.command()
@click.option('--data', type=str, default='ml-1m', help='Select Dataset')
@click.option('--seed', type=int, default=0, help='Set random seed')
@click.option('--reg', type=bool, default=True, help='Use TDR if True')
@click.option('--unmask', type=bool, default=False, help='Use unmask scheme if True')
@click.option('--ut', type=int, default=0, help='Number of unmasking top items')
@click.option('--ur', type=int, default=0, help='Number of unmasking random items')
@click.option('--ep', type=int, default=200, help='Number of total epoch')
@click.option('--reclen', type=int, default=30, help='Number of epoch with reccommendation loss')
@click.option('--dim', type=int, default=32, help='Number of latent factors')
@click.option('--cpu', type=bool, default=False, help='Use CPU while TDR')
@click.option('--dut', type=float, default=0, help='Change on the number of unmasking top items per epoch')
@click.option('--dur', type=float, default=0, help='Change on the number of unmasking random items per epoch')
@click.option('--rbs', type=int, default=0, help='Number of rows in mini batch')
@click.option('--cbs', type=int, default=0, help='Number of columns in mini batch')
def main(data, seed, reg, unmask, ut, ur, ep, reclen, dim, cpu, dut, dur, rbs, cbs):
    set_seed(seed)
    device = DEVICE
    # set hyperparameters
    config = {
        'lr': 1e-3,
        'decay': 1e-4,
        'latent_dim': dim,
        'batch_size': 4096,
        'epochs': ep,
        'ks': [5, 10],
        'trn_neg': 4,
        'test_neg': 99
    }
    print(config)

    torch.multiprocessing.set_sharing_strategy('file_system')

    # load data
    trn_path = f'../data/{data}/train'
    vld_path = f'../data/{data}/validation'
    test_path = f'../data/{data}/test'

    train_data, test_data, user_num, item_num, train_mat = load_all(trn_path, test_path)

    train_dataset = BPRData(
        train_data, item_num, train_mat, config['trn_neg'], True)
    test_dataset = BPRData(
        test_data, item_num, train_mat, 0, False)
    train_loader = DataLoader(train_dataset,
                              batch_size=config['batch_size'], shuffle=True,
                              num_workers=4)
    test_loader = DataLoader(test_dataset,
                             batch_size=config['test_neg'] + 1,
                             shuffle=False, num_workers=0)

    # define model and optimizer
    model = BPR(user_num, item_num, config['latent_dim'])
    model.to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=config['lr'], weight_decay=config['decay'])

    # show dataset stat
    print('user:', user_num, '  item:', item_num, '  tr len:', len(train_data))

    header = f'Epoch | '
    for k in config['ks']:
        header += f'Recall@{k:2d} NDCG@{k:2d} C@{k:2d} G@{k:2d} E@{k:2d} | '
    header += f'Duration (sec)'
    print(header)

    # obtain items in training set and ground truth items from data
    train_data = [[] for _ in range(user_num)]
    gt = []
    with open(test_path, 'r') as fd:
        line = fd.readline()
        while line != None and line != '':
            arr = line.split('\t')
            gt.append(eval(arr[0])[1])
            line = fd.readline()
    init_time = time.time()

    # start model training
    for epoch in range(1, config['epochs']+1):
        model.train()
        start_time = time.time()
        train_loader.dataset.ng_sample()

        if epoch == 1:
            num_batch = 0
        # train with recommendation loss
        if epoch <= reclen:
            for user, item_i, item_j in train_loader:
                if epoch == 1:
                    for u, i in zip(user, item_i):
                        train_data[u].append(i)
                user = user.to(device)
                item_i = item_i.to(device)
                item_j = item_j.to(device)

                # recommendation loss
                model.zero_grad()
                prediction_i, prediction_j = model(user, item_i, item_j)
                rec_loss = - (prediction_i - prediction_j).sigmoid().log().sum()
                rec_loss.backward()
                optimizer.step()

                if epoch == 1:
                    num_batch += 1

        # move model to cpu if option cpu is true
        if epoch == reclen and cpu:
            device = torch.device('cpu')
            model = model.to(device)
            optimizer_to(optimizer, device)

        # train with diversity regularizer
        if reg and epoch > reclen:
            # top-k inference
            k = config['ks'][1]            
            
            if rbs == 0:
                row_batch_size = user_num
            else:
                row_batch_size = rbs
            row_batch = list_chunk(torch.randperm(user_num).tolist(), row_batch_size)
            if cbs == 0:
                col_batch_size = item_num
            else:
                col_batch_size = cbs
            col_batch = list_chunk(torch.randperm(item_num).tolist(), col_batch_size)
            
            # calculate number of unmasking items for each mini batch
            bk = math.ceil(k / len(col_batch))
            bur = math.ceil(max(ur + int((epoch-reclen-1)*dur), 0) / len(col_batch))
            but = math.ceil(max(ut + int((epoch-reclen-1)*dut), 0) / len(col_batch))

            for rb in row_batch:
                for cb in col_batch:
                    # inference top-k recommendation lists
                    model.zero_grad()
                    scores = []
                    items = torch.LongTensor(cb).to(device)
                    for u in rb:
                        u = torch.tensor([u]).to(device)
                        score, _ = model(u, items, items)
                        scores.append(score)
                    scores = torch.stack(scores)
                    scores = torch.softmax(scores, dim=1)

                    # unmasking mechanism
                    if unmask:
                        k_ = len(cb) - (bk+but)
                    else:
                        k_ = len(cb) - bk
                    mask_idx = torch.topk(-scores, k=k_)[1]  # mask index for being filled 0
                    if unmask:
                        for u in range(len(rb)):
                            idx = torch.randperm(mask_idx.shape[1])
                            mask_idx[u] = mask_idx[u][idx]
                        if bur > 0:
                            mask_idx = mask_idx[:, :-bur]

                    mask = torch.zeros(size=scores.shape, dtype=torch.bool)
                    mask[torch.arange(mask.size(0)).unsqueeze(1), mask_idx] = True
                    topk_scores = scores.masked_fill(mask.to(device), 0)

                    # coverage regularizer
                    scores_sum = torch.sum(topk_scores, dim=0, keepdim=False)
                    epsilon = 0.01
                    scores_sum += epsilon
                    d_loss = -torch.sum(torch.log(scores_sum))

                    # skewness regularizer
                    topk_scores = torch.topk(scores, k=k)[0]
                    norm_scores = topk_scores / torch.sum(topk_scores, dim=1, keepdim=True)
                    e_loss = torch.sum(torch.sum(norm_scores * torch.log(norm_scores), dim=1))

                    # sum of losses
                    regularizations = d_loss + e_loss
                    regularizations.backward()
                    optimizer.step()
        
        # evaluate metrics
        model.eval()
        HRs, NDCGs, coverages, Gs, Es = [], [], [], [], []
        for k in config['ks']:
            rec_list = make_rec_list(model, k, user_num, item_num, train_data, device)
            HR, NDCG, coverage, giny, entropy = metrics_from_list(rec_list, item_num, gt)
            HRs.append(HR)
            NDCGs.append(NDCG)
            coverages.append(coverage)
            Gs.append(giny)
            Es.append(entropy)

        epoch_elapsed_time = time.time() - start_time
        total_elapsed_time = time.time() - init_time

        # print evaluated metrics to console
        content = f'{epoch:6d} | '
        for hr, ndcg, coverage, g, e in zip(HRs, NDCGs, coverages, Gs, Es):
            content += f'{hr:.4f} {ndcg:.4f} {coverage:.4f} {g:.4f} {e:.4f} | '
        content += f'{epoch_elapsed_time:.1f} {total_elapsed_time:.1f}'
        print(content)

if __name__ == '__main__':
    main()

