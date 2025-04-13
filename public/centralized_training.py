from collections import OrderedDict
from typing import List, Tuple, Dict, Callable, Optional, Any, get_type_hints
import inspect
import warnings
import math
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split
import sklearn.metrics as sklearn_metrics
from sklearn.model_selection import train_test_split


class Model(nn.Module):
    def __init__(self, params_dict={}) -> None:
        super(Model, self).__init__()

        if params_dict == {}:
            raise RuntimeError('Some model parameters must be provided.')
            
        #else:
        #    self.set_params_dict(params_dict)
                
            
        # list of linear layers
        linears = params_dict['linears']
        self.linears = nn.ModuleList()
        
        # list of activation functions
        actvs = params_dict['actvs']
        self.actvs = nn.ModuleList()
        
        if len(self.linears) != len(self.actvs):
            warning.warn('Missmatch between linear layers and respective activation functions.')
            
        # stack layers
        for k, (i,j) in enumerate(linears):
            self.linears.append(nn.Linear(i,j))
            if actvs[k] == 'ELU':
                activation = nn.ELU(alpha=-2)
                print('ELU alpha = -2')
            else: activation = getattr(nn, actvs[k])()
            self.actvs.append(activation)


    def forward(self, x):
        # Convert x from torch.float64 to torch.float32
        x = x.to(torch.float32)

        # pass input through the network
        for k in range(len(self.linears)):
            x = self.linears[k](x)
            x = self.actvs[k](x)

        x = x.squeeze()
        return x


def train(train_loader, model, DEVICE, lr, epochs, criterion=None, optimizer=None,
          metrics=['accuracy_score', 'f1_score', 'recall_score', 'precision_score', 'roc_auc_score'], verbose=False):

    if criterion == None:
        criterion = nn.BCELoss()
    if optimizer == None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) #, weight_decay=0.001)

    model.to(DEVICE)
    criterion.to(DEVICE)

    for epoch in range(epochs):
        model.train()
        for tuples, labels in train_loader:
            tuples, labels = tuples.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(tuples)

            loss = criterion(outputs.squeeze(), labels)
            loss.backward()

            optimizer.step()


    results = test(train_loader, model, DEVICE, criterion, metrics, verbose)
            
    return model, results


def test(test_loader, model, DEVICE, criterion=None, 
         metrics=['accuracy_score', 'f1_score', 'recall_score', 'precision_score', 'roc_auc_score'],
         verbose=False, threshold=0.5, **kwargs):
    
    if criterion == None:
        criterion = nn.BCELoss()

    model.to(DEVICE)
    criterion.to(DEVICE)
    model.eval()
    test_loader = DataLoader(test_loader.dataset, batch_size=len(test_loader.dataset), shuffle=True)
    
    with torch.no_grad():
        loss = 0.0
        for tuples, labels in test_loader:
            tuples, labels = tuples.to(DEVICE), labels.to(DEVICE)

            outputs = model(tuples)
            loss += criterion(outputs.squeeze(), labels).item()
            predictions = (outputs > threshold).float()


    results = dict(zip(metrics, np.zeros(len(metrics))))
    results = calculate_metrics(results, labels, predictions, outputs)
    
    if verbose:
        print(f'Loss: {loss:.4f}')
        for m in results.keys():
            print(f'{m}: {results[m]:.4f}')
            
    return results


def calculate_metrics(results, labels, predictions, outputs):
    for metric in results.keys():
        try:
            getattr(sklearn_metrics, metric)
        except:
            raise NotImplementedError(
                'Unknown metric or not implemented by scikit learn.'
            )
        if metric != 'roc_auc_score':
            results[metric] = getattr(sklearn_metrics, metric)(labels.numpy(), predictions.numpy())
        else:
            try:
                results[metric] = getattr(sklearn_metrics, metric)(labels.numpy(), outputs.numpy())
            except ValueError as e:
                if e == 'Only one class present in y_true. ROC AUC score is not defined in that case.':
                    results[metric] = 1.0
                    print(e)
    return results


def parse_list_int(value):
    try:
        # Split the input string by commas and convert each part to an integer
        linears = [int(x) for x in value.split(',')]
        return linears
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid value for linears. Please provide a comma-separated list of integers.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COPMODE centralized training')
    parser.add_argument('-s', '--seed', type=int, default=4)
    parser.add_argument('-sh', '--shadow', type=int, default=2000)
    parser.add_argument('-p', '--train_size', type=float, default=0.8)
    parser.add_argument('-lr', '--lr', type=float, default=1e-4)
    parser.add_argument('-e', '--epochs', type=int, default=25)
    parser.add_argument('-l', '--linears', type=parse_list_int, default=[100], help='Split ints by commas')
    parser.add_argument('-a', '--actvs', type=str, default='ReLU')
    parser.add_argument('-o', '--outfunc', type=str, default='Sigmoid')
    
    args = parser.parse_args()
    # missing example