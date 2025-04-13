import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import scipy.stats as st
from centralized_training import Model
from sklearn.model_selection import train_test_split
import pickle
from torch.utils.data import DataLoader, TensorDataset
from IPython.utils.capture import capture_output
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from scipy.special import rel_entr
from report_results import get_interval

def malicious_neuron_detection(bef, cur, neurons, total, attack_round=5, alpha=0.1556):
    neurons = list(range(-neurons,0))
    total = list(np.arange(-total,0)) # total = num neurons

    div = [kl_divergence(bef[row,:], cur[row,:], len(total)) for row in total]
    malicious = [(d >= alpha)+0 for d in div]
    tpr = np.sum([(row in neurons and malicious[row] == 1)+0 for row in total])/len(neurons)
    fpr = np.sum([(not(row in neurons) and malicious[row] == 1)+0 for row in total])/(len(total)- len(neurons))
    missed = np.sum([(row in neurons and malicious[row] == 0)+0 for row in total])/len(neurons)
    index = [row for row in total if malicious[row] == 1]
    
    return tpr, fpr, missed, index

def replace(bef, cur, layers, index):
    # replace malicious neurons weights (and bias) in cur by their weights in the previous round (bef)
    out = copy.deepcopy(cur)
    for layer in layers:
        for i in index:
            try: # weights
                out[layer][i,:] = bef[layer][i,:]
            except: # bias
                out[layer][i] = bef[layer][i]
    return out

def kl_divergence(x, y, n=128, bins=5):
    a = min(min(np.array(x)), min(np.array(y)))
    b = max(max(np.array(x)), max(np.array(y)))
    bef, _, _ = plt.hist(np.array(x), bins=bins, range=(a, b))
    plt.close()
    cur, _, _ = plt.hist(np.array(y), bins=bins, range=(a, b))
    plt.close()
    
    bef = bef/n
    cur = cur/n
    return min(sum(rel_entr(bef, cur)), sum(rel_entr(cur, bef)))

def optimal_threshold(bef, cur, neurons, total, attack_round=5):
    alphas = np.linspace(start=0.05, stop=0.5, num=10)
    results = dict(zip(alphas, np.repeat([0.0, 0.0, 0.0], len(alphas))))
    best_alpha, best_result = 0.0, [-np.inf, 1.0, 0.0] # tpr, fpr, missed
    neurons, total = list(range(-neurons,0)), list(np.arange(-total,0)) # total = numneurons 
    div = [kl_divergence(bef[row,:], cur[row,:], len(total)) for row in total]
    for alpha in alphas:
        malicious = [(d >= alpha)+0 for d in div]
        tpr = np.sum([(row in neurons and malicious[row] == 1)+0 for row in total])/len(neurons)
        fpr = np.sum([(not(row in neurons) and malicious[row] == 1)+0 for row in total])/(len(total)- len(neurons))
        missed = np.sum([(row in neurons and malicious[row] == 0)+0 for row in total])/len(neurons)
        
        #tpr, fpr, missed, _ = malicious_neuron_detection(bef=bef, cur=cur, neurons=neurons, total=total,
        #                                                 attack_round=attack_round, alpha=alpha)
        results[alpha] = [tpr, fpr, missed]
        if results[alpha][0]/(results[alpha][1]+results[alpha][2]) > best_result[0]/(best_result[1]+best_result[2]):
            best_result = results[alpha]
            best_alpha = alpha
        if best_result[0]/(best_result[1]+best_result[2]) == 1.0: break
    return best_alpha, best_result

def automate_WADM(dataset='celeba_models', attack='ELU', attribute='515', model='federated_5009', total_neurons=512, N=1):
    benign_path = dataset + '/' + attack + '_WADM' + '/' + model
    malign_path = dataset + '/' + attack + '_attack_' + attribute + '/' + model
    tpr, fpr, missed, alpha = [], [], [], []
    
    for i in range(32):
        benign_model = torch.load(benign_path + f'_{i}.pth')
        malign_model = torch.load(malign_path + f'_{i}.pth')
        layers = [l for l in benign_model.keys() if '0' in l.split('.') or '1' in l.split('.')]
        
        best_alpha, best_result = optimal_threshold(bef=benign_model[layers[2]], cur=malign_model[layers[2]],
                                                    neurons=N, total=total_neurons)
        
        tpr.append(best_result[0])
        fpr.append(best_result[1])
        missed.append(best_result[2])
        alpha.append(best_alpha)
    print('TPR', get_interval(tpr))
    print('FPR', get_interval(fpr))
    print('Missed Attacks', get_interval(missed))
    #print(tpr)
    #print(fpr)
    #print(missed)
    #print(alpha)