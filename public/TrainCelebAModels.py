import pandas as pd
import pickle
import numpy as np
import sys
import os
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


# load and prepare dataset
def get_data_loaders(seed, oversample=None, shadow=5000):
    inputs, labels = torch.load('../data_celebA/celeba_inputs.pth'), torch.load('../data_celebA/celeba_labels.pth') # labels is pd.series
    torch.manual_seed(seed)
    n = len(inputs) - shadow # 5009 = shadow size
    splits = [int(n/10) for _ in range(10)] # 10 randomly sampled clients
    splits.append(shadow)
    idx = torch.split(torch.arange(0, len(inputs)), splits)
    local_training_datasets, local_testing_datasets = [], []

    for i in range(len(idx[:-1])):
        local_inputs, local_labels = inputs[idx[i]], labels[list(idx[i])]
        train_idx, test_idx = torch.split(idx[i], [int(0.8*len(idx[i])), len(idx[i]) - int(0.8*len(idx[i]))], dim=0)
        local_train_loader = DataLoader(TensorDataset(inputs[train_idx.int()], torch.tensor(labels[list(train_idx.int())].values, dtype=torch.float32)), batch_size=100)
        local_test_loader = DataLoader(TensorDataset(inputs[test_idx.int()], torch.tensor(labels[list(test_idx.int())].values, dtype=torch.float32)), batch_size=100)
        local_training_datasets.append(local_train_loader)
        local_testing_datasets.append(local_test_loader)
        
    shadow_data = pd.DataFrame(inputs[idx[-1].int()].numpy())
    shadow_data['Label'] = labels[list(idx[-1].int())]
    
    return local_training_datasets, local_testing_datasets, shadow_data # dataloder, dataloader, pd.df


def get_datasets(seed, oversample=None, shadow=5000):
    inputs, labels = torch.load('../data_celebA/celeba_inputs.pth'), torch.load('../data_celebA/celeba_labels.pth') # labels is pd.series
    torch.manual_seed(seed)
    n = len(inputs) - shadow # 5009 = shadow size
    splits = [int(n/10) for _ in range(10)] # 10 randomly sampled clients
    splits.append(shadow)
    idx = torch.split(torch.arange(0, len(inputs)), splits)
    local_training_datasets, local_testing_datasets = [], []

    for i in range(len(idx[:-1])):
        local_inputs, local_labels = inputs[idx[i]], labels[list(idx[i])]
        train_idx, test_idx = torch.split(idx[i], [int(0.8*len(idx[i])), len(idx[i]) - int(0.8*len(idx[i]))], dim=0)
        local_train_loader = DataLoader(TensorDataset(inputs[train_idx.int()], torch.tensor(labels[list(train_idx.int())].values, dtype=torch.float32)))
        local_test_loader = DataLoader(TensorDataset(inputs[test_idx.int()], torch.tensor(labels[list(test_idx.int())].values, dtype=torch.float32)))
        df = pd.DataFrame(local_train_loader.dataset.tensors[0].numpy())
        df['Label'] = local_train_loader.dataset.tensors[1].numpy()
        local_training_datasets.append(df)
        df = pd.DataFrame(local_test_loader.dataset.tensors[0].numpy())
        df['Label'] = local_test_loader.dataset.tensors[1].numpy()
        local_testing_datasets.append(df)
        
    shadow_data = pd.DataFrame(inputs[idx[-1].int()].numpy())
    shadow_data['Label'] = labels[list(idx[-1].int())]
    
    return pd.concat(local_training_datasets), pd.concat(local_testing_datasets), shadow_data


def train_model(seed, model_path, n_clients, rounds, local_training_datasets, local_testing_datasets, params_dict):
    torch.manual_seed(seed)
    DEVICE = torch.device('cpu')

    fl_training = Simulation(seed=seed, strategy='FedAVG',
                             n_clients=n_clients,
                             clients_training_data=local_training_datasets,
                             clients_test_data=local_testing_datasets,
                             params_dict=params_dict,
                             hyperparameters={'lr': 0.001, 'epochs': 5, 'DEVICE': DEVICE, 'verbose':False},
                             save_path=model_path)
    fl_training.start(DEVICE, rounds=rounds, global_model=None)
    
    
def evaluate(model_path, local_datasets, params_dict, threshold=0.5):
    # avg of local performance
    DEVICE = torch.device('cpu')
    fl_model = Model(params_dict)
    fl_model.load_state_dict(torch.load(model_path))
    avg_results = {'accuracy_score': 0.0, 'f1_score': 0.0, 'recall_score': 0.0, 'precision_score': 0.0, 'roc_auc_score': 0.0}
    total = np.sum([len(local_loader.dataset) for local_loader in local_datasets])

    for c, local_loader in enumerate(local_datasets):
        print(f'\nClient {c}')
        rs_test = test(local_loader, fl_model, DEVICE, verbose=True, threshold=threshold)
        size = len(local_loader.dataset)
        for metric in list(avg_results.keys()):
            avg_results[metric] += (size/total)*rs_test[metric]
    return avg_results


def save_performance(model_path, local_datasets, params_dict, metric, n_round, save_path, threshold, n=None): # for black-box strategies
    DEVICE = torch.device('cpu')
    fl_model = Model(params_dict)
    fl_model.load_state_dict(torch.load(model_path))
    
    # create file if save_path does not exist
    # create pd.df per metric with as much rows as len(local_datasets) and 3 columns (round_0, round_1, round_2 - malicious)
    if not(os.path.exists(save_path)):
        pd.DataFrame({'Round_0': np.zeros(len(local_datasets)), 'Round_1': np.zeros(len(local_datasets)),
                      f'Round_2_{n}': np.zeros(len(local_datasets))}, index=[f'client_{c}' for c in range(len(local_datasets))]).to_csv(save_path, index=True)
    df = pd.read_csv(save_path, index_col=0)
    
    if not(f'Round_2_{n}' in df.columns): df[f'Round_2_{n}'] = np.zeros(len(local_datasets))
    
    for c, local_loader in enumerate(local_datasets):
        performance = test(local_loader, fl_model, DEVICE, verbose=False, threshold=threshold)[metric]
        if n_round == 2: df.loc[f'client_{c}'][f'Round_{n_round}_{n}'] = performance
        else: df.loc[f'client_{c}'][f'Round_{n_round}'] = performance
    df.to_csv(save_path)
    
    
def recover_model(seed, attack_path, model_path, n_clients, rounds, local_training_datasets, local_testing_datasets, params_dict):
    torch.manual_seed(seed)
    DEVICE = torch.device('cpu')

    fl_training = Simulation(seed=seed, strategy='FedAVG',
                             n_clients=n_clients,
                             clients_training_data=local_training_datasets,
                             clients_test_data=local_testing_datasets,
                             params_dict=params_dict,
                             hyperparameters={'lr': 0.001, 'epochs': 5, 'DEVICE': DEVICE, 'verbose':False},
                             save_path=model_path)
    
    attack_model = Model(params_dict)
    attack_model.load_state_dict(torch.load(attack_path))
    fl_training.start(DEVICE, rounds=rounds, global_model=attack_model)
    
    
def automate_save_performance(actv='ReLU', threshold=0.5, n=515): # save performance monitorization
    np.random.seed(42)
    seeds = np.random.randint(low=0, high=1234, size=32)
    params_dict = {'linears': [(518, 1024), (1024, 512), (512, 1)], # simple model is better, one hot encoding of categorical features
                   'actvs': [actv, actv, 'Sigmoid'],
                   'loss': 'BCELoss', 'optimizer': 'Adam'}
    
    for i, seed in enumerate(seeds):
        local_training_datasets, local_testing_datasets, shadow_data = get_data_loaders(seed=seed, oversample=None, shadow=5009)
        if not(os.path.exists(f'celeba_models/random_{actv}_federated_5009_{i}.pth')):
            train_model(seed, model_path=f'celeba_models/random_{actv}_federated_5009_{i}.pth', n_clients=10, rounds=0,
                        local_training_datasets=local_training_datasets, local_testing_datasets=local_testing_datasets, params_dict=params_dict)
            save_performance(model_path=f'celeba_models/random_{actv}_federated_5009_{i}.pth',
                             local_datasets=local_training_datasets, params_dict=params_dict,
                             metric='accuracy_score', n_round=0, save_path=f'celeba_models/{actv}_monitorization_accuracy_{i}.csv', threshold=threshold)
            save_performance(model_path=f'celeba_models/random_{actv}_federated_5009_{i}.pth',
                             local_datasets=local_training_datasets, params_dict=params_dict,
                             metric='roc_auc_score', n_round=0, save_path=f'celeba_models/{actv}_monitorization_auc_{i}.csv', threshold=threshold)
            save_performance(model_path=f'celeba_models/{actv}_WADM/federated_5009_{i}.pth',
                             local_datasets=local_training_datasets, params_dict=params_dict,
                             metric='accuracy_score', n_round=1, save_path=f'celeba_models/{actv}_monitorization_accuracy_{i}.csv', threshold=threshold)
            save_performance(model_path=f'celeba_models/{actv}_WADM/federated_5009_{i}.pth',
                             local_datasets=local_training_datasets, params_dict=params_dict,
                             metric='roc_auc_score', n_round=1, save_path=f'celeba_models/{actv}_monitorization_auc_{i}.csv', threshold=threshold)
            sys.exit('should be skiped')
        
        
        save_performance(model_path=f'celeba_models/{actv}_attack_{n}/federated_5009_{i}.pth',
                         local_datasets=local_training_datasets, params_dict=params_dict,
                         metric='accuracy_score', n_round=2, n=n, save_path=f'celeba_models/{actv}_monitorization_accuracy_{i}.csv', threshold=threshold)
        save_performance(model_path=f'celeba_models/{actv}_attack_{n}/federated_5009_{i}.pth',
                         local_datasets=local_training_datasets, params_dict=params_dict,
                         metric='roc_auc_score', n_round=2, n=n, save_path=f'celeba_models/{actv}_monitorization_auc_{i}.csv', threshold=threshold)
    

def automate(): # train 32 models to compute confidence intervals
    np.random.seed(42)
    seeds = np.random.randint(low=0, high=1234, size=32)
    train_auc, train_acc, test_auc, test_acc = [], [], [], []
    
    params_dict = {'linears': [(518, 1024), (1024, 512), (512, 1)], # simple model is better, one hot encoding of categorical features
                   'actvs': ['ELU', 'ELU', 'Sigmoid'],
                   'loss': 'BCELoss', 'optimizer': 'Adam'}
    
    for i, seed in enumerate(seeds):
        print(f'-------------------- Model {i} --------------------')
        with capture_output() as output:
            local_training_datasets, local_testing_datasets, shadow_data = get_data_loaders(seed=seed,
                                                                                            oversample=None,
                                                                                            shadow=5009)

            train_model(seed=seed,
                        model_path=f'celeba_models/ReLU_WADM/federated_5009_{i}.pth',
                        n_clients=10, rounds=1,
                        local_training_datasets=local_training_datasets,
                        local_testing_datasets=local_training_datasets,
                        params_dict=params_dict) # REMOVE OUTPUT AND WARNIGNS FROM THIS FUNCTION

            train_results = evaluate(model_path=f'celeba_models/ReLU_WADM/federated_5009_{i}.pth',
                                     local_datasets=local_training_datasets,
                                     params_dict=params_dict,
                                     threshold=0.2)
            test_results = evaluate(model_path=f'celeba_models/ReLU_WADM/federated_5009_{i}.pth',
                                    local_datasets=local_testing_datasets,
                                    params_dict=params_dict,
                                    threshold=0.2)
        print('Train results:\n', train_results)
        print('Test results:\n', test_results)
        train_auc.append(train_results['roc_auc_score'])
        test_auc.append(test_results['roc_auc_score'])
        train_acc.append(train_results['accuracy_score'])
        test_acc.append(test_results['accuracy_score'])
    print(get_interval(train_auc))
    print(get_interval(test_auc))
    print(get_interval(train_acc))
    print(get_interval(test_acc))
    

def automate_recovery(attribute=515): # train 32 models to compute confidence intervals
    np.random.seed(42)
    seeds = np.random.randint(low=0, high=1234, size=32)
    train_auc, train_acc, test_auc, test_acc = [], [], [], []
    
    params_dict = {'linears': [(518, 1024), (1024, 512), (512, 1)], # simple model is better, one hot encoding of categorical features
                   'actvs': ['ReLU', 'ReLU', 'Sigmoid'],
                   'loss': 'BCELoss', 'optimizer': 'Adam'}
    
    for i, seed in enumerate(seeds):
        print(f'-------------------- Model {i} --------------------')
        with capture_output() as output:
            local_training_datasets, local_testing_datasets, shadow_data = get_data_loaders(seed=seed,
                                                                                            oversample=None,
                                                                                            shadow=5009)

            recover_model(seed=seed,
                        attack_path=f'celeba_models/ReLU_attack_{attribute}/federated_5009_{i}.pth',
                        model_path=f'celeba_models/ReLU_recovery_{attribute}/federated_5009_{i}.pth',
                        n_clients=10, rounds=1,
                        local_training_datasets=local_training_datasets,
                        local_testing_datasets=local_training_datasets,
                        params_dict=params_dict)

            train_results = evaluate(model_path=f'celeba_models/ReLU_recovery_{attribute}/federated_5009_{i}.pth',
                                     local_datasets=local_training_datasets,
                                     params_dict=params_dict)
            test_results = evaluate(model_path=f'celeba_models/ReLU_recovery_{attribute}/federated_5009_{i}.pth',
                                    local_datasets=local_testing_datasets,
                                    params_dict=params_dict)
        print('Train results:\n', train_results)
        print('Test results:\n', test_results)
        train_auc.append(train_results['roc_auc_score'])
        test_auc.append(test_results['roc_auc_score'])
        train_acc.append(train_results['accuracy_score'])
        test_acc.append(test_results['accuracy_score'])
    print(get_interval(train_auc))
    print(get_interval(test_auc))
    print(get_interval(train_acc))
    print(get_interval(test_acc))
        