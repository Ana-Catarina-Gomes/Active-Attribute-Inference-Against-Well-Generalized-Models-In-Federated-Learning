# Black-box strategies
from TrainCelebAModels import automate_save_performance
import pandas as pd
import pickle
import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.stats as st
from centralized_training import Model, train, test
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from myFL import Simulation
from IPython.utils.capture import capture_output
import warnings
warnings.filterwarnings('ignore')


def get_interval(v, conf=0.90):
    #with warnings.catch_warnings():
        #warnings.filterwarnings("ignore", category=RuntimeWarning)
        
    interval = st.norm.interval(confidence=conf, loc=np.mean(v), scale=np.sqrt(np.var(v)))
    interval = (np.round(interval[0]*100, 2), np.round(interval[1]*100, 2))
    if np.all([np.isnan(interval[0]), np.isnan(interval[1])]):
        interval = (np.mean(v)*100, np.mean(v)*100)
                        
    return adjust_confidence_interval(interval, N=len(v))


def adjust_confidence_interval(interval, N=30):
    a, b = interval
    new_error_margin = (b - a) / (2 * np.sqrt(N))
    mean = (a + b) / 2
    adjusted_interval = (np.round(mean - new_error_margin, 2), np.round(mean + new_error_margin, 2))
    return adjusted_interval

# remove function not needed
def evaluate(model_path, local_datasets, params_dict):
    # avg of local performance
    DEVICE = torch.device('cpu')
    fl_model = Model(params_dict)
    fl_model.load_state_dict(torch.load(model_path))
    avg_results = {'accuracy_score': 0.0, 'f1_score': 0.0, 'recall_score': 0.0, 'precision_score': 0.0, 'roc_auc_score': 0.0}
    total = np.sum([len(local_loader.dataset) for local_loader in local_datasets])

    for c, local_loader in enumerate(local_datasets):
        print(f'\nClient {c}')
        rs_test = test(local_loader, fl_model, DEVICE, verbose=True)
        size = len(local_loader.dataset)
        for metric in list(avg_results.keys()):
            avg_results[metric] += (size/total)*rs_test[metric]
    return avg_results


class SPCDetection:
    def __init__(self):
        self.Pmin = 1.0
        self.Smin = np.inf

    def update(self, y, n, verbose=True):
        # y = prop of bernoulli var (acc), 1-y = prop of bernoulli var (0-1 loss)
        # eqv a ver se existe dif significativa de performance entre rounds, não entre rounds consecutivas

        # p is 1-acc (E(miss predictions)) of round i, s is std of miss predictions of round i
        # s^2 = E(miss predictions^2) - p^2 = p - p^2 = p(1-p)
            
        p = abs(1-y)
        s = (p * (1 - p)/n) ** 0.5

        # Check process status
        if p + s < self.Pmin + 2 * self.Smin:
            status = "In-control"
        elif p + s >= self.Pmin + 3 * self.Smin:
            status = "Out-control"
        else:
            status = "Warning Level"
            
        # Update Pmin and Smin
        if p + s != 0 and p + s < self.Pmin + self.Smin:
            self.Pmin = p
            self.Smin = s

        if verbose:
            print(f"Current (p, s) = ({p}, {s})\n(Pmin, Smin) = ({self.Pmin}, {self.Smin})")
            
        return status
    
    
def BADAcc_test(info_path, sizes, attack_round=5, n_clients=93, n=515):   
    tprs, fprs, tprs_warning, fprs_warning, misseds = [], [], [], [], []

    for j in range(32):
        path = info_path + f'_{j}.csv'
        #print(f'Iteration {i}')
        info = pd.read_csv(path).drop('Unnamed: 0', axis=1, inplace=False)
        info.drop([col for col in info.columns if col.startswith(f'Round_{attack_round}') and col != f'Round_{attack_round}_{n}'], axis=1, inplace=True)
        #info[f'Round_{attack_round}_{n}'] = np.ones(info.shape[0])*0.5

        TPR, false_alarm, missed, TPR_warning, FPR_warning = 0, 0, 0, 0, 0
        for c in range(0, n_clients): # over clients
            #print(f'Client {c}')
            spc = SPCDetection()
            for i in range(1, info.shape[1]): # over rounds
                status = spc.update(info.iloc[c,i], n=sizes[j][c], verbose=False)
                #print(f'Round {i+1}: {status}')
                if status == 'Out-control' and i == attack_round:
                    TPR += 1
                    break
                elif status == 'Warning Level' and i == attack_round:
                    TPR_warning += 1
                elif status == 'Out-control' and i < attack_round:
                    false_alarm += 1
                    break
                elif status == 'Out-control' and i > attack_round:
                    missed += 1
                    break
                elif status == 'Warning Level' and i != attack_round:
                    FPR_warning += 1

        if info.shape[1]-1 == attack_round: missed = n_clients - TPR
        TPR /= n_clients
        false_alarm /= n_clients
        missed /= n_clients
        TPR_warning /= n_clients
        FPR_warning /= n_clients
        tprs.append(TPR)
        fprs.append(false_alarm)
        misseds.append(missed)
        tprs_warning.append(TPR_warning)
        fprs_warning.append(FPR_warning)
    
    return tprs, fprs, misseds


# detect attack through avg model weights or auc
class AvgDetector:
    def __init__(self, init):
        self.window = [init]

    def update(self, new, alpha):
        self.window.append(new)
        #dif = abs(self.window[1] - self.window[0]) # ideally positive
        dif = -(self.window[1] - self.window[0])
        if dif >= alpha:
            status = 'Out-control'
        else:
            status = 'In-control'
        self.window = [new]
        return status
    
def test_thresholds(info_path, attack_round=5, a=0.00005, b=0.11, n_clients=93, n=515):
    alphas = np.linspace(start=a, stop=b, num=100)
    T, F, M = [], [], []
    for alpha in alphas:
        TPR, false_alarm, missed = [], [], []
        for j in range(32):
            TPR.append(0)
            false_alarm.append(0)
            missed.append(0)
            
            path = info_path + f'_{j}.csv'
            #print(f'Iteration {i}')
            info = pd.read_csv(path).drop('Unnamed: 0', axis=1, inplace=False)
            info.drop([col for col in info.columns if col.startswith(f'Round_{attack_round}') and col != f'Round_{attack_round}_{n}'], axis=1, inplace=True)
            
            for c in range(n_clients):# over clients
                detector = AvgDetector(info.iloc[c,0])
                for i in range(1, info.shape[1]): # over rounds
                    status = detector.update(info.iloc[c,i], alpha)
                    #print(f'Round {i+1}: {status}')
                    if status == 'Out-control' and i == attack_round:
                        TPR[-1] += 1
                        break
                    if status == 'Out-control' and i < attack_round:
                        false_alarm[-1] += 1
                        break
                    if i == attack_round:
                        missed[-1] += 1
                        break
            
            TPR[-1] = TPR[-1]/n_clients
            false_alarm[-1] = false_alarm[-1]/n_clients
            missed[-1] = missed[-1]/n_clients
        T.append(get_interval(TPR))
        F.append(get_interval(false_alarm))
        M.append(get_interval(missed))

    return {'alphas': alphas, 'T': T, 'F': F, 'M': M}

def find_optimal(results):
    best = 0
    alpha = []
    for i in range(len(results['alphas'])):
        if np.mean(results['T'][i])/(np.mean(results['F'][i]) + np.mean(results['M'][i])) > best:
            best = np.mean(results['T'][i])/(np.mean(results['F'][i]) + np.mean(results['M'][i]))
            alpha = [i]
        elif np.mean(results['T'][i])/(np.mean(results['F'][i]) + np.mean(results['M'][i])) == best:
            best = np.mean(results['T'][i])/(np.mean(results['F'][i]) + np.mean(results['M'][i]))
            alpha.append(i)

    if len(alpha) > 1:
        alpha = alpha[np.argmin([np.mean(results['F'][i]) for i in alpha])]
    else: alpha = alpha[0]
    return alpha



def automate_BADAcc(feature, n=515, info_path='celeba_models/', sizes=[19759*np.ones(10) for _ in range(32)], attack_round=2, n_clients=10):
    # feature stands for the sensitive attribute target by attacks that modify these models
    print(f'Feature {feature}:')
    print('MIA2AIA')
    tprs, fprs, misseds = BADAcc_test(info_path=info_path+'ReLU_monitorization_accuracy', sizes=sizes, attack_round=attack_round, n_clients=n_clients, n=n)
    print(f'TPR = {get_interval(tprs)} | FPR = {get_interval(fprs)} | Missed = {get_interval(misseds)}')
    #print(tprs)
    
    print('AAI')
    tprs, fprs, misseds = BADAcc_test(info_path=info_path+'ELU_monitorization_accuracy', sizes=sizes, attack_round=attack_round, n_clients=n_clients, n=n)
    print(f'TPR = {get_interval(tprs)} | FPR = {get_interval(fprs)} | Missed = {get_interval(misseds)}')
    #print(tprs)
    
def automate_BADAUC(feature, n=515, info_path='celeba_models/', attack_round=2, n_clients=10):
    print(f'Feature {feature}:')
    print('MIA2AIA')
    results = test_thresholds(info_path+'ReLU_monitorization_auc', attack_round=attack_round, a=0.005, b=0.15, n_clients=n_clients, n=n)
    alpha = find_optimal(results)
    print('TPR', results['T'][alpha], '| FPR =', results['F'][alpha], '| Missed =', results['M'][alpha], 'alpha =', results['alphas'][alpha])
    
    print('AAI')
    results = test_thresholds(info_path+'ELU_monitorization_auc', attack_round=attack_round, a=0.005, b=0.15, n_clients=n_clients, n=n)
    alpha = find_optimal(results)
    print('TPR', results['T'][alpha], '| FPR =', results['F'][alpha], '| Missed =', results['M'][alpha], 'alpha =', results['alphas'][alpha])