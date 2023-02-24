'''
Top-K Diversity Regularizer

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

'''

import random

import numpy as np
from numpy.core.numeric import indices
import pandas as pd
import scipy.sparse as sp
import torch
from torch._C import Value
import torch.utils.data as data
from tqdm import tqdm

CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA else 'cpu')


def set_seed(seed):
    '''
    Set pytorch random seed as seed.
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if CUDA:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def hit(gt_item, pred_items):
    '''
    Check whether given recommendation list hits or not.
    gt_item : ground truth item
    pred_items : list of recommended items
    '''
    if gt_item in pred_items:
        return 1
    return 0

def ndcg(gt_item, pred_items):
    '''
    Calculate nDCG
    gt_item : ground truth item
    pred_items : list of recommended items
    '''
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index+2))
    return 0

def metrics_from_list(R, item_num, gt):
    '''
    Calculate all metrics from recommendation list
    return average Hit Ratio, nDCG, coverage, gini index, entropy of R
    R : list of recommendation lists
    item_num : number of items in dataset
    gt : list of ground truth items
    '''
    HR, NDCG = [], []
    rec_items = []
    cnt = [0 for i in range(item_num)]
    for r, gt_item in zip(R, gt):
        HR.append(hit(gt_item, r))
        NDCG.append(ndcg(gt_item, r))
        rec_items += r
        for i in r:
            cnt[i] += 1
    coverage = len(set(rec_items))/item_num
    giny = 0
    cnt.sort()
    height, area = 0, 0
    for c in cnt:
        height += c
        area += height-c/2
    fair_area = height*item_num/2
    giny = (fair_area-area)/fair_area
    a = torch.FloatTensor(cnt)
    a/=sum(a)
    entropy = torch.distributions.Categorical(probs=a).entropy()
    return np.mean(HR), np.mean(NDCG), coverage, giny, entropy

def make_rec_list(model, top_k, user_num, item_num, train_data, device=DEVICE):
    '''
    Build recommendation lists from the model
    model : recommendation model
    top_k : length of a recommendation list
    user_num : number of users in dataset
    item_num : number of items in dataset
    train_data : lists of items that a user interacted in training dataset
    device : device where the model mounted on
    '''
    rtn = []
    for u in range(user_num):
        items = torch.tensor(list(set(range(item_num))-set(train_data[u]))).to(device)
        u = torch.tensor([u]).to(device)
        score, _ = model(u, items, items)
        _, indices = torch.topk(score, top_k)
        recommends = torch.take(items, indices).cpu().numpy().tolist()
        rtn.append(recommends)
    return rtn

def load_all(trn_path, test_neg, test_num=100):
    """ We load all the three file here to save time in each epoch. """
    '''
    Load dataset from given path
    trn_path : path of training dataset
    test_neg : path of test dataset
    '''
    train_data = pd.read_csv(
        trn_path,
        sep='\t', header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1

    train_data = train_data.values.tolist()

    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    test_data = []
    with open(test_neg, 'r') as fd:
        line = fd.readline()
        while line != None and line != '':
            arr = line.split('\t')
            u = eval(arr[0])[0]
            test_data.append([u, eval(arr[0])[1]])
            for i in arr[1:]:
                test_data.append([u, int(i)])
            line = fd.readline()
    return train_data, test_data, user_num, item_num, train_mat


class BPRData(data.Dataset):
    def __init__(self, features,
                 num_item, train_mat=None, num_ng=0, is_training=None):
        super(BPRData, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
        features : data
        num_item : number of items
        train_mat : interaction matrix
        num_ng : number of negative samples
        is_training : is model training
        """
        self.features = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training

    def ng_sample(self):
        '''
        Sample negative items for BPR
        '''
        assert self.is_training, 'no need to sampling when testing'

        self.features_fill = []
        for x in self.features:
            u, i = x[0], x[1]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_fill.append([u, i, j])

    def __len__(self):
        '''
        Number of instances.
        '''
        return self.num_ng * len(self.features) if \
            self.is_training else len(self.features)

    def __getitem__(self, idx):
        '''
        Grab an instance.
        '''
        features = self.features_fill if \
            self.is_training else self.features

        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2] if \
            self.is_training else features[idx][1]
        return user, item_i, item_j

def optimizer_to(optim, device):
    '''
    Move optimizer to target device
    optim : optimizer
    device : target device
    '''
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)