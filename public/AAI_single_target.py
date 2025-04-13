import sys
import copy
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


class SingleCustomMSE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y_pred, y_true, weight):
        ctx.save_for_backward(y_pred, y_true)
        ctx.weight = weight  # Store the weight tensor in the context
        loss = torch.mean((y_pred - y_true)**2)
        return loss

    @staticmethod
    def backward(ctx, grad_output): 
        y_pred, y_true = ctx.saved_tensors
        weight = ctx.weight  # Retrieve the weight tensor from the context
        grad_input = 2 * (y_pred - y_true) * grad_output
        grad_input = (weight.float()*grad_input.squeeze(-1)).unsqueeze(-1)
        w = weight.size(0)
        return grad_input/w, -grad_input/w, None
        
        
# class that defines adversarial model
class Classifier(nn.Module):
    def __init__(self, n_inputs, numneurons, n_outputs, trainable=True):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(n_inputs, numneurons)
        self.fc2 = nn.Linear(numneurons, n_outputs)

        # Set requires_grad to False for non-trainable parameters
        for param in self.fc1.parameters():
            param.requires_grad = trainable

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        f = nn.ELU(alpha=-2)
        x = f(x)
        fc2 = self.fc2(x)
        return fc2

    def set_parameters(self, parameters):
        layers = [l for l in parameters.keys() if '0' in l.split('.') or '1' in l.split('.')]
        parameters = {k: v for k,v in parameters.items() if k in layers}

        parameters['fc1.weight'] = parameters.pop(layers[0])
        parameters['fc1.bias'] = parameters.pop(layers[1])

        n = self.fc2.out_features
        neuron_weights = parameters[layers[2]][-n:, :]  # choose last neuron
        neuron_bias = parameters[layers[3]][-n:]
        
        parameters.pop(layers[2])
        parameters.pop(layers[3])
        parameters['fc2.weight'] = neuron_weights
        parameters['fc2.bias'] = neuron_bias
        
        self.load_state_dict(parameters, strict=True)


def elu_prime(s, alpha=-2):
    if s > 0:
        return 1
    else:
        return alpha*np.exp(s)
    
def sigmoid_prime(conf):
    if conf == 0: conf = np.exp(-100)
    if conf == 1: conf = 1 - np.exp(-100)
    return conf*(1-conf)


def BCELoss_prime(conf, label, w=[1.0, 1.0]):
    if label == 0:
        return max(conf, np.exp(-100))*max(1-conf, np.exp(-100))/max((1-conf), np.exp(-100)) # w[0]/(1-conf) # changed sign
    if label == 1:
        return -max(conf, np.exp(-100))*max(1-conf, np.exp(-100))/max(conf, np.exp(-100)) # -w[1]/conf


def get_attribute_values(targets, unknown):
    # assume binary attribute
    positive = targets.copy()
    negative = targets.copy()

    if len(targets.shape) == 1:
        positive[unknown] = 1.0
        negative[unknown] = 0.0
    else:
        n = targets.shape[0]
        positive.iloc[:, unknown] = np.repeat(1.0, n)
        negative.iloc[:, unknown] = np.repeat(0.0, n)
    
    return positive, negative

        
# create adversarial model
def adv_model_init(numneurons, global_model=None, n_target=1, n_input=59):
    # global_model.parameters() - exchange only model parameters, not model class
    adv_model = Classifier(n_input, numneurons, n_target)
        
    if global_model != None:
        adv_model.set_parameters(global_model)
    
    return adv_model
    
    
def AAI_adversary(adv_model, shadow, positives, negatives, learning_rate, epochs, seed, remove=1):
    DEVICE = torch.device('cpu')
    N = 1
    
    x_train, x_test = train_test_split(shadow.iloc[:, :-remove], train_size=0.8, random_state=seed) # shadow is a pd df
    # 1/0 - target, -1 - non target
    rep = np.repeat(-1.0, N)
    rep1, rep0 = rep, rep
    y_train = torch.tensor([rep] * x_train.shape[0])
    y_test = torch.tensor([rep] * x_test.shape[0])

    x_target1, y_target1, x_target0, y_target0 = [], [], [], []
    for i in range(N):
        rep1 = np.repeat(-1.0, N)
        rep1[i] = 1.0
        if N > 1:
            tuples = torch.tensor([positives.iloc[i,:].values.astype(np.float32)] * 1)
        else:
            tuples = torch.tensor([positives.values.astype(np.float32)] * 1)
        labels = torch.tensor([rep1] * 1)
        x_target1.append(tuples)
        y_target1.append(labels)

        rep0 = np.repeat(-1.0, N)
        rep0[i] = -0.01
        if N > 1:
            tuples = torch.tensor([negatives.iloc[i,:].values.astype(np.float32)] * 1)
        else:
            tuples = torch.tensor([negatives.values.astype(np.float32)] * 1)
        labels = torch.tensor([rep0] * 1)
        x_target0.append(tuples)
        y_target0.append(labels)
    
    x_target1 = torch.stack(x_target1, dim=1)
    x_target1 = x_target1.view(-1, x_target1.size(-1))
    x_target0 = torch.stack(x_target0, dim=1)
    x_target0 = x_target0.view(-1, x_target0.size(-1))
    y_target1 = torch.stack(y_target1, dim=1)
    y_target1 = y_target1.view(-1, y_target1.size(-1))
    y_target0 = torch.stack(y_target0, dim=1)
    y_target0 = y_target0.view(-1, y_target0.size(-1))
    x_train_tensor = torch.tensor(x_train.values.astype(np.float32))
    x_train = torch.cat((x_train_tensor, x_target1, x_target0))
    y_train = torch.cat((y_train, y_target1, y_target0))

    x_test = torch.tensor(x_test.values.astype(np.float32))
    print('Train and test shapes:')
    print(f'{x_train.shape} | {y_train.shape}')
    print(f'{x_test.shape} | {y_test.shape}')

    # train
    adv_model.to(DEVICE)
    adv_model.train()
    optimizer= torch.optim.Adam(adv_model.parameters(), lr=learning_rate)
    
    weight = torch.cat((torch.tensor(2*np.ones(x_train.shape[0]-2*N)),
                        torch.tensor(20000*np.ones(N)),
                        torch.tensor(20000*np.ones(N))))
        
    for e in range(epochs):
        fc2 = adv_model(x_train)
        loss = SingleCustomMSE.apply(fc2, 400*y_train.float(), weight)
        loss.backward() 
        optimizer.step()              # make the updates for each parameter
        optimizer.zero_grad()         # a clean up step for PyTorch

    # Validation
    with torch.no_grad():
        adv_model.eval()
        tns, fps,  tps, fns = 0.0, 0.0, 0.0, 0.0

        for k in range(N):
            fc2 = adv_model(x_train_tensor)
            predictions = [-2*np.exp(fc2[i,k]) == 0 if fc2[i,k] <= 0.0 else False for i in range(fc2.shape[0])] # when this is true it's a TN
            n = x_train_tensor.shape[0]
            tn = torch.sum(torch.tensor(predictions))
            fp = n - tn
            print(f'Training ({e} epochs): TNR = {tn/n}, FPR = {fp/n}')

            tn, fp = 0, 0
            fc2 = adv_model(x_test)
            predictions = [-2*np.exp(fc2[i,k]) == 0 if fc2[i,k] <= 0.0 else False for i in range(fc2.shape[0])] # when this is true it's a TN
            n = x_test.shape[0]
            tn = torch.sum(torch.tensor(predictions))
            fp = n - tn
            avg = torch.mean(fc2)

            tp, fn = 0, 0
            if N == 1: fc2 = adv_model(torch.tensor([positives.values], dtype=torch.float32))
            else: fc2 = adv_model(torch.tensor([positives.iloc[k,:].values], dtype=torch.float32))
            if fc2[:,k] > 0.0: tp += 1
            else: fn += 1
            print(fc2)

            if N ==1: fc2 = adv_model(torch.tensor([negatives.values], dtype=torch.float32))
            else: fc2 = adv_model(torch.tensor([negatives.iloc[k,:].values], dtype=torch.float32))
            if fc2[:,k] <= 0.0 and -2*np.exp(fc2[:,k]) != 0: tp += 1
            else: fn += 1
            print(fc2)

            tns += tn/n
            fps += fp/n
            tps += tp/2
            fns += fn/2
             
        print(f'Validation: TNR = {tns/N}, FPR = {fps/N}, TPR = {tps/N}, FNR = {fns/N}\n')
        
    return adv_model  
    
    
def theoretical_eval(N, target, data_train, unknown, adv_model, flip, remove):
    # theoretical evaluation - domain of neuron activation
    print('Inference phase')
    tp, fn = 0.0, 0.0
    
    for i in range(N):
        with torch.no_grad():
            fc2 = adv_model(torch.tensor(np.array(target.iloc[i, :-remove].values).astype(np.float32)).unsqueeze(0))
            true = target.iloc[i, unknown]
                
        if fc2[:,i] > 0.0: predictions = 1
        else: predictions = -2*np.exp(fc2[:,i])
            
        if predictions < 0:
            if flip: pred = 1.0
            else: pred = 0.0
            tp += (pred==true)+0
            fn += (pred!=true)+0
        elif predictions == 1:
            if flip: pred = 0.0
            else: pred = 1.0
            tp += (pred==true)+0
            fn += (pred!=true)+0
        else:
            fn += 1
            
    tp /= N
    fn /= N
            
    print('Attack report:')
    print(f'TPR = {tp} | FNR = {fn}')
    
    tn, fp = 0.0, 0.0
    for i in range(N):
        try: df = data_train.drop(indexes[i], axis=0, inplace=False)
        except: df = data_train
        
        neg = df.shape[0]
        test = torch.tensor(np.array(df.iloc[:, :-remove].values).astype(np.float32))
        with torch.no_grad(): fc2 = adv_model(test)
        predictions = [-2*np.exp(fc2[j,i]) == 0 if fc2[j,i] <= 0.0 else False for j in range(fc2.shape[0])]
            
        tn += torch.sum(torch.tensor(predictions))/neg
        fp += (neg - torch.sum(torch.tensor(predictions)))/neg
            
    tn /= N
    fp /= N
    print(f'TNR = {tn} | FPR = {fp}')
    return tp, fn, tn, fp


def WADM_eval(N, model_w, global_model, adv_model, alpha, total_neurons, save_path, mitigate=False):
    print('\n\n============== Changing weights ==============')
    # change main task model weights (model_w is from the previous round)
    adv_w = adv_model.state_dict()
    layers = [l for l in model_w.keys() if '0' in l.split('.') or '1' in l.split('.')]
    bef = copy.deepcopy(model_w)
    global_model[layers[0]] = adv_w['fc1.weight']
    global_model[layers[1]] = adv_w['fc1.bias']

    if N > 1:
        for i in range(1, N+1):
            global_model[layers[2]][-i,:] = adv_w['fc2.weight'][(i-1),:] # neuron weights
            global_model[layers[3]][-i] = adv_w['fc2.bias'][(i-1)] # neuron bias

    else:
        global_model[layers[2]][-1,:] = adv_w['fc2.weight'][0,:] # neuron weights
        global_model[layers[3]][-1] = adv_w['fc2.bias'][0] # neuron bias
    torch.save(global_model, save_path) # save attacked model
            
    print('\n\n============== Malicious neuron detection ==============')
    tpr, fpr, missed, index = malicious_neuron_detection(bef=bef[layers[2]], cur=global_model[layers[2]],
                              neurons=N, total=total_neurons, alpha=alpha) # total_neurons = params_dict['linears'][1][1]
    
    if not(mitigate): return tpr, fpr, missed, index

    model_w = replace(bef=bef, cur=global_model, layers=layers[2:], index=index)
    if N > 1: # switch malicious by benign neurons
        for i in range(1, N+1):
            adv_w['fc2.weight'][(i-1),:] = model_w[layers[2]][-i,:] # neuron weights
            adv_w['fc2.bias'][(i-1)] = model_w[layers[3]][-i] # neuron bias
    else:
        adv_w['fc2.weight'][0,:] = model_w[layers[2]][-1,:] # neuron weights
        adv_w['fc2.bias'][0] = model_w[layers[3]][-1] # neuron bias
    adv_model.set_parameters(model_w)
    adv_w = adv_model.state_dict()
    return tpr, fpr, missed, index, adv_w # evaluate attack performance with these adv_w model weights


def realistic_eval(N, data_train, adv_model, global_model, neurons, remove, mode='attack'):
    DEVICE = torch.device('cpu')
    criterion = nn.BCELoss().to(DEVICE)
    
    n_input = global_model['linears.0.weight'].shape[1]
    numneurons = global_model['linears.0.weight'].shape[0]
    secondFC = global_model['linears.1.weight'].shape[0]
    params_dict = {'linears': [(n_input, numneurons), (numneurons, secondFC), (secondFC, 1)],
                   'actvs': ['ELU', 'ELU', 'Sigmoid'],
                   'loss': 'BCELoss', 'optimizer': 'Adam'}
    
    model = Model(params_dict)
    for i in range(N):
        neuron = neurons[i]
        if target_label == 1.0:
            global_model['linears.2.weight'][:, neuron]= -abs(global_model['linears.2.weight'][:, neuron])
            # switch sign to macth BCELoss_prime sign
        if target_label == 0.0:
            global_model['linears.2.weight'][:, neuron]= abs(global_model['linears.2.weight'][:, neuron])
            # switch sign to macth BCELoss_prime sign

    # change weights
    layers = [l for l in global_model.keys() if '0' in l.split('.') or '1' in l.split('.')]
    global_model[layers[0]] = adv_model.state_dict()['fc1.weight']
    global_model[layers[1]] = adv_model.state_dict()['fc1.bias']
    
    if mode == 'attack':
        global_model[layers[2]][-1,:] = adv_model.state_dict()['fc2.weight'][0,:] # assuming single target attacks
        global_model[layers[3]][-1] = adv_model.state_dict()['fc2.bias'][0]

    if mode == 'mitigate':
        # WADM mitigation phase
        adv_w = adv_model.state_dict()
        adv_w['fc2.weight'][0,:] = global_model[layers[2]][-1,:] # assuming single target attacks
        adv_w['fc2.bias'][0] = global_model[layers[3]][-1]
        adv_model = adv_model_init(numneurons, global_model=None, n_target=1, n_input=n_input)
        adv_model.load_state_dict(adv_w)

    model.load_state_dict(global_model)
    model.to(DEVICE)
    adv_model.to(DEVICE)
    
    gradients = []
    weights = model.state_dict()['linears.2.weight']
    tuples = torch.tensor(np.array(data_train.iloc[:, :-remove].values).astype(np.float32)).to(DEVICE)
    labels = torch.tensor(data_train.iloc[:, -remove].values, dtype=torch.float32).to(DEVICE)
    confs = model(tuples)
    fc2 = adv_model(tuples).detach()
    grad = sum([BCELoss_prime(conf.item(), y.item())*elu_prime(fc2[i, 0])*weights[:, neurons[0]] for i, conf, y in zip(range(data_train.shape[0]), confs, labels)])
    gradients = [grad.item()/data_train.shape[0]]
    
    return gradients
    
    
# conduct and evaluate AAI experiment
def exp(numneurons, n_input, seed, unknown, lr, shadow, epochs, target, indexes, model, data_train, evaluation='theoretical', remove=1, model_w=None, alpha=None, save_path=None):
    
    # for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    DEVICE = torch.device('cpu')
    
    N = 1
    positives, negatives = get_attribute_values(targets=target.iloc[:, :-remove], unknown=unknown)

    # label flipping - true if most frequent value is 0.0
    flip = list(dict(shadow[shadow.columns[unknown]].value_counts(sort=True, ascending=False)).keys())[0] == 0.0

    global_model = torch.load(model)
    adv_model = adv_model_init(numneurons, global_model=global_model, n_target=1, n_input=n_input)

    if flip:
        adv_model = AAI_adversary(adv_model, shadow, negatives, positives, learning_rate=lr, epochs=epochs, seed=seed, remove=remove)
    else:
        adv_model = AAI_adversary(adv_model, shadow, positives, negatives, learning_rate=lr, epochs=epochs, seed=seed, remove=remove)
    
    if evaluation == 'theoretical':
        tp, fn, tn, fp = theoretical_eval(N, target, data_train, unknown, adv_model, flip, remove)
        return tp, fn, tn, fp, adv_model
    if evaluation == 'realistic': # assuming single-target
        gradient = realistic_eval(N, data_train, adv_model, global_model, neurons=[-1], remove=remove, mode='attack')
        return gradient[0], adv_model
    if evaluation == 'WADM_detection':
        tp, fp, missed, index = WADM_eval(N, model_w, global_model, adv_model, alpha, total_neurons=int(numneurons/2),
                                          save_path=save_path, mitigate=False) # total_neurons = nb of neurons from the 2nd FC layer
        return tp, fp, missed, index, adv_model
    if evaluation == 'WADM_mitigation_theoretical':
        tp, fp, missed, index, adv_w = WADM_eval(N, model_w, global_model, adv_model, alpha, total_neurons=int(numneurons/2),
                                                 save_path=save_path, mitigate=True)
        adv_model = adv_model_init(numneurons, global_model=global_model, n_target=N, n_input=n_input)
        adv_model.load_state_dict(adv_w)
        adv_model.to(DEVICE)
        tp, fn, tn, fp = theoretical_eval(N, target, data_train, unknown, adv_model, flip, remove)
        return tp, fn, tn, fp, adv_model
    if evaluation == 'WADM_mitigation_realistic':
        tp, fp, missed, index, adv_w = WADM_eval(N, model_w, global_model, adv_model, alpha, total_neurons=int(numneurons/2),
                                                 save_path=save_path, mitigate=True)
        adv_model = adv_model_init(numneurons, global_model=global_model, n_target=N, n_input=n_input)
        adv_model.load_state_dict(adv_w)
        adv_model.to(DEVICE)
        gradient = realistic_eval(N, data_train, adv_model, global_model, neurons=[-1], remove=remove, mode='mitigate')
        return gradient[0], adv_model
