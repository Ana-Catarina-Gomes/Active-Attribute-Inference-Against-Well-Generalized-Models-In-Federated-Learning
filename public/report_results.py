import numpy as np
import pandas as pd
import torch
from sklearn import metrics
import matplotlib.pyplot as plt
import scipy.stats as st
import warnings


def get_auc(pos, neg, max_fpr, sigmoid=True):
    labels = list(np.ones(len(pos))) + list(np.zeros(len(neg)))
    if sigmoid:
        scores = list(torch.sigmoid(torch.tensor(pos, dtype=torch.float64))) + list(torch.sigmoid(torch.tensor(neg, dtype=torch.float64)))
    else:
        scores = list(pos) + list(neg)
    
    return metrics.roc_auc_score(labels, scores, max_fpr=max_fpr)

def get_auc_multi(pos, neg, max_fpr, sigmoid=True):
    n = len(pos)
    labels = list(np.tile([1, 0], n).reshape((n, 2))) + list(np.tile([0, 1], n).reshape((n, 2)))
    pos = [np.array(sublist) for sublist in pos]
    neg = [np.array(sublist) for sublist in neg]
    
    if sigmoid:
        scores = list(torch.sigmoid(torch.tensor(pos))) + list(torch.sigmoid(torch.tensor(neg)))
    else:
        scores = list(pos) + list(neg)
    
    return metrics.roc_auc_score(labels, scores, max_fpr=max_fpr, multi_class='ovo')

def avg_auc(pos, neg, non, max_fpr, sigmoid=True):
    return (get_auc(pos, neg, max_fpr) + get_auc(pos, non, max_fpr) + get_auc(non, neg, max_fpr))/3

def auc_fprs(pos, neg, multi=False, sigmoid=False):
    fprs = np.linspace(0.0, 1.0, 10)[1:]
    if not(multi):
        aucs = [get_auc(pos=pos, neg=neg, max_fpr=fpr, sigmoid=sigmoid) for fpr in fprs]
    else:
        aucs = [get_auc_multi(pos=pos, neg=neg, max_fpr=fpr, sigmoid=sigmoid) for fpr in fprs]
    return aucs

def auc_report(Grads, pos_idx=15, neg_idx=0, begin=0, switch=False, multi=False, sigmoid=False):
    fprs = np.linspace(0.0, 1.0, 10)[1:]
    aucs = dict(zip(fprs, [[] for _ in range(10)]))
    for grads in Grads:
        if not(switch):
            auc = auc_fprs(pos=grads[begin:pos_idx], neg=grads[pos_idx:neg_idx], multi=multi, sigmoid=sigmoid)
        else:
            auc = auc_fprs(neg=grads[begin:pos_idx], pos=grads[pos_idx:neg_idx], multi=multi, sigmoid=sigmoid)
        for i, fpr in enumerate(fprs):
            aucs[fpr].append(auc[i])
        
    # plot([get_interval(auc) for _,auc in aucs.items()])
    return dict(zip(fprs, [get_interval(auc) for _,auc in aucs.items()]))

def adjust_confidence_interval(interval, N=30):
    a, b = interval
    new_error_margin = (b - a) / (2 * np.sqrt(N))
    mean = (a + b) / 2
    adjusted_interval = (np.round(mean - new_error_margin, 2), np.round(mean + new_error_margin, 2))
    return adjusted_interval


def get_interval(v, conf=0.90):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        interval = st.norm.interval(confidence=conf, loc=np.mean(v), scale=np.sqrt(np.var(v)))
        interval = (np.round(interval[0]*100, 2), np.round(interval[1]*100, 2))
        if np.all([np.isnan(interval[0]), np.isnan(interval[1])]):
            interval = (np.mean(v)*100, np.mean(v)*100)
                        
    return adjust_confidence_interval(interval, len(v))


def error_margin(interval, N=30):
    a, b = interval
    return (b - a) / (2 * np.sqrt(N))

def report_mia2aia(mia2aia_grads, least_frequent_label=1): # report restricted ROC AUC (practical evalaution)
    # maximize precision of the least frequent label, meaning the adv would be most interested in predicting the least frequent label
    
    if least_frequent_label == 1: # positive samples are from the least frequent label for ROC AUC calculation purposes
        def get_score(grads):
            return [torch.sigmoid(torch.tensor(g[1])).item() if g[0] < g[1] else torch.sigmoid(torch.tensor(-g[0])).item() for g in grads]
        labels = list(np.zeros(8)) + list(np.ones(8))
    else:
        def get_score(grads):
            return [torch.sigmoid(torch.tensor(g[0])).item() if g[1] < g[0] else torch.sigmoid(torch.tensor(-g[1])).item() for g in grads]
        labels = list(np.ones(8)) + list(np.zeros(8))
        
    members, non_members, n = [], [], len(mia2aia_grads)
    for i in range(n):
        members.append(metrics.roc_auc_score(labels, get_score(mia2aia_grads[i])[:int(n/2)], max_fpr=0.25))
        non_members.append(metrics.roc_auc_score(labels, get_score(mia2aia_grads[i])[int(n/2):], max_fpr=0.25))

    print(get_interval(members))
    print(get_interval(non_members))
    
def report_aai(aai_grads, least_frequent_label=1): # report restricted ROC AUC (practical evalaution)
    # maximize precision of the least frequent label, meaning the adv would be most interested in predicting the least frequent label
    def get_score(grads): # get_score is the same due to label flipping on aai experiments
        return [torch.sigmoid(torch.tensor(-g, dtype=torch.float64)).item() for g in grads]
        
    if least_frequent_label == 1: # positive samples are from the least frequent label for ROC AUC calculation purposes
        labels = list(np.zeros(8)) + list(np.ones(8))
    else:
        labels = list(np.ones(8)) + list(np.zeros(8))
        
    members, non_members, n = [], [], len(aai_grads)
    for i in range(n):
        members.append(metrics.roc_auc_score(labels, get_score(aai_grads[i])[:int(n/2)], max_fpr=0.25))
        non_members.append(metrics.roc_auc_score(labels, get_score(aai_grads[i])[int(n/2):], max_fpr=0.25))

    print(get_interval(members))
    print(get_interval(non_members))