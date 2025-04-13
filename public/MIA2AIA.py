import sys
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from centralized_training import Model
from sklearn.model_selection import train_test_split
import pickle
import scipy.stats as st
from torch.utils.data import DataLoader, TensorDataset
from IPython.utils.capture import capture_output
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from scipy.special import rel_entr


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
        x = F.relu(x)
        fc2 = self.fc2(x)
        x = torch.sigmoid(fc2)
        probs = F.softmax(x, dim=1)
        return x, probs, fc2

    def set_parameters(self, parameters):
        layers = [l for l in parameters.keys() if '0' in l.split('.') or '1' in l.split('.')]
        parameters = {k: v for k,v in parameters.items() if k in layers}

        parameters['fc1.weight'] = parameters.pop(layers[0])
        parameters['fc1.bias'] = parameters.pop(layers[1])

        n = self.fc2.out_features -1
        if n == 1:
            neuron_weights = parameters[layers[2]][-n, :]  # choose last neuron
            neuron_bias = parameters[layers[3]][-n]
            duplicated_weights = torch.stack([neuron_weights, parameters[layers[2]][-1, :]], dim=0)
            duplicated_bias = torch.stack([torch.unsqueeze(neuron_bias, 0), torch.unsqueeze(parameters[layers[3]][-1], 0)], dim=0)
        else:
            neuron_weights = parameters[layers[2]][-n:, :]  # choose last neuron
            neuron_bias = parameters[layers[3]][-n:]
            duplicated_weights = torch.cat([parameters[layers[2]][-1:, :], neuron_weights], dim=0)
            duplicated_bias = torch.cat([parameters[layers[3]][-1:], neuron_bias])

        parameters.pop(layers[2])
        parameters.pop(layers[3])
        parameters['fc2.weight'] = duplicated_weights
        parameters['fc2.bias'] = duplicated_bias.squeeze()
        
        self.load_state_dict(parameters, strict=True)
        
        
# create logistic regression
def adv_model_init(numneurons, global_model=None, n_target=1, n_input=59): # global_model.parameters() - exchange only model parameters, not model class
    adv_model = Classifier(n_input, numneurons, n_target+1)
        
    if global_model != None:
        adv_model.set_parameters(global_model)
    
    return adv_model


def get_attribute_values(targets, unknown): # assume binary attribute
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

def my_tpr_tnr(prediction, truth): # negatives = 0, positives = 1
    confusion_vector = prediction / truth

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives / (true_positives + false_negatives), true_negatives / (true_negatives + false_positives), (true_positives + true_negatives) / (true_negatives + false_negatives + true_positives + false_positives)


def relu_prime(s):
    return (s > 0).item()

def sigmoid_prime(conf):
    if conf == 0: conf = np.exp(-100)
    if conf == 1: conf = 1 - np.exp(-100)
    return conf*(1-conf)

def BCELoss_prime(conf, label, w=[1.0, 1.0]): # np.exp(-100) for numerical stabiltiy
    if label == 0:
        return max(conf, np.exp(-100))*max(1-conf, np.exp(-100))/max((1-conf), np.exp(-100))
    if label == 1:
        return -max(conf, np.exp(-100))*max(1-conf, np.exp(-100))/max(conf, np.exp(-100))


def AMI_adversary(adv_model, shadow, targets, learning_rate, epochs, seed, remove=2):
    DEVICE = torch.device('cpu')
    N = targets.shape[0]
    
    x_train, x_test = train_test_split(shadow.iloc[:, :-remove], train_size=0.8, random_state=seed) # shadow is a pd df
    # 0 - non target, 1 - target
    y_train = np.zeros((x_train.shape[0], N+1))
    y_train[:,0] = 1 # first neuron is one for non targets
    y_test = np.zeros((x_test.shape[0], N+1))
    y_test[:,0] = 1

    x_target, y_target = [], []
    for i in range(N):
        x_target.append(targets.iloc[i, :])
        y = np.zeros(N+1)
        y[i+1] = 1 # neuron i+1 is one only for target sample
        y_target.append(y)

    x_target = torch.tensor(np.array(x_target).astype(np.float32))
    
    x_train = torch.tensor(x_train.values.astype(np.float32))
    x_train = torch.cat((x_train, x_target))
    y_train = torch.tensor(y_train)
    y_train = torch.cat((y_train, torch.tensor(y_target)))
    
    x_test = torch.tensor(x_test.values.astype(np.float32))
    x_test = torch.cat((x_test, x_target))
    y_test = torch.tensor(y_test)
    y_test = torch.cat((y_test, torch.tensor(y_target)))
    
    print('Train and test shapes:')
    print(f'{x_train.shape} | {y_train.shape}')
    print(f'{x_test.shape} | {y_test.shape}')
    
    custom_weight = np.zeros(N+1)
    custom_weight[0] = 0.1
    for i in range(1,N+1):
        custom_weight[i] = 25000/N
        
    #custom_weight = np.array([0.1, 25000]) # give greater importance to target sample 25000/0.1
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(custom_weight, dtype=torch.float).to(DEVICE))
    optimizer= torch.optim.Adam(adv_model.parameters(), lr=learning_rate)

    # train
    adv_model.to(DEVICE)
    adv_model.train()

    for e in range(epochs):
        out, probs, fc2 = adv_model(x_train)
        loss = criterion(out, y_train.float())
        loss.backward() 
        optimizer.step()              # make the updates for each parameter
        optimizer.zero_grad()         # a clean up step for PyTorch

    # training performance
        tprs, fprs, accs = 0.0, 0.0, 0.0
        for i in range(1, fc2.shape[1]):
            predictions = fc2[:, i] > 0
            tpr_train, tnr_train, acc = my_tpr_tnr(predictions, y_train[:, i])
            tprs += tpr_train
            fprs += 1 - tnr_train
            accs += acc
        
    print(f'Epoch {e}: TPR {tprs/N} | FPR {fprs/N} | Accuracy {accs/N}')

    # Validation
    adv_model.eval()
    out, probs, fc2 = adv_model(x_test)
    tprs, fprs, accs = 0.0, 0.0, 0.0
    for i in range(1, fc2.shape[1]):
        predictions = fc2[:, i] > 0
        tpr_train, tnr_train, acc = my_tpr_tnr(predictions, y_test[:, i])
        tprs += tpr_train
        fprs += 1 - tnr_train
        accs += acc
    print(f'Valiation: TPR {tprs/N} | FPR {fprs/N} | Accuracy {accs/N}')

    return adv_model



def exp(numneurons, n_input, seed, unknown, lr, shadow, epochs, target, indexes, model, data_train, type_attack='single', remove=1, evaluation='theoretical', imputation=True, model_w=None, alpha=None, save_path=None):

    # for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    DEVICE = torch.device('cpu')
    
    #if type_attack == 'single':
    #    N = 1
    #    positives, negatives = get_attribute_values(targets=target.iloc[:, :-remove], unknown=unknown)
    #    samples = [positives, negatives]
    #if type_attack == 'multi':
    N = target.shape[0]
    positives, negatives = get_attribute_values(targets=target.iloc[:, :-remove], unknown=unknown)
    samples, targets = [], []
    for t in range(N):
        #print(f'Check target sample real sensitive value = {target.iloc[t, unknown]}')
        samples.append(positives.iloc[t, :])
        samples.append(negatives.iloc[t, :])

    samples = pd.DataFrame(np.array(samples))
        
    global_model = torch.load(model)
    adv_model = adv_model_init(numneurons, global_model=global_model, n_target=N*2, n_input=n_input)
    adv_model = AMI_adversary(adv_model, shadow, samples, learning_rate=lr, epochs=epochs, seed=seed, remove=remove)

    if evaluation == 'theoretical':
        tp, fn, tn, fp = theoretical_eval(N=N, target=target.iloc[:, :-remove], data_train=data_train.iloc[:, :-remove], unknown=unknown, adv_model=adv_model, shadow=shadow, imputation=imputation)
        return tp, fn, tn, fp, adv_model
    if evaluation == 'realistic': # assuming single-target, not tested for multi-target attacks
        gradient = realistic_eval(N=N, data_train=data_train, adv_model=adv_model, global_model=global_model,
                                  neurons=[-2, -1], remove=remove, mode='attack')
        return gradient[0], adv_model
    if evaluation == 'WADM_detection':
        tp, fp, missed, index = WADM_eval(N, model_w, global_model, adv_model, alpha, total_neurons=int(numneurons/2),
                                       save_path=save_path, mitigate=False) # total_neurons = nb of neurons from the 2nd FC layer
        return tp, fp, missed, index, adv_model
    if evaluation == 'WADM_mitigation_theoretical':
        tp, fp, missed, index, adv_w = WADM_eval(N, model_w, global_model, adv_model, alpha, total_neurons=int(numneurons/2),
                                                 save_path=save_path, mitigate=True)
        adv_model = adv_model_init(numneurons, global_model=global_model, n_target=N*2, n_input=n_input)
        adv_model.load_state_dict(adv_w)
        adv_model.to(DEVICE)
        tp, fn, tn, fp = theoretical_eval(N=N, target=target.iloc[:, :-remove], data_train=data_train.iloc[:, :-remove], unknown=unknown, adv_model=adv_model, shadow=shadow, imputation=imputation)
        return tp, fn, tn, fp, adv_model
    if evaluation == 'WADM_mitigation_realistic':
        tp, fp, missed, index, adv_w = WADM_eval(N, model_w, global_model, adv_model, alpha, total_neurons=int(numneurons/2),
                                                 save_path=save_path, mitigate=True)
        adv_model = adv_model_init(numneurons, global_model=global_model, n_target=N*2, n_input=n_input)
        adv_model.load_state_dict(adv_w)
        adv_model.to(DEVICE)
        gradient = realistic_eval(N=N, data_train=data_train, adv_model=adv_model, global_model=global_model,
                                  neurons=[-2, -1], remove=remove, mode='mitigate')
        return gradient[0], adv_model


def theoretical_eval(N, target, data_train, unknown, adv_model, shadow, imputation=True):
    # evaluation
    print('Inference phase')
    print('--------------------------------------')
    tp, fn, i = 0.0, 0.0, 1
    for t in range(N):
        _, _, fc2 = adv_model(torch.tensor(np.array(target.iloc[t, :].values).astype(np.float32)).unsqueeze(0))
        if (fc2[:, i] > 0).item() == 1 and (fc2[:, i+1] > 0).item() == 1:
            print('Inconclusive') # INCONCLUSIVE -> MOST COMMON VALUE
            if imputation:
                prediction = list(dict(shadow.iloc[:, unknown].value_counts(sort=True, ascending=False)).keys())[0]
                print(f'Prediction = {prediction}')
                tp += (prediction == target.iloc[t, unknown])+0
                fn += (prediction != target.iloc[t, unknown])+0
            else: fn += 1
        else:
            if target.iloc[t, unknown] == 1.0: # contabilizar apenas a tp qd só se ativa o neurónio correto, caso contrário é fp
                tp += (fc2[:, i] > 0).item() # tp in case correct neuron is activated
                fn += (fc2[:, i] < 0).item()
            elif target.iloc[t, unknown] == 0.0:
                tp += (fc2[:, i+1] > 0).item() # tp in case correct neuron is activated
                fn += (fc2[:, i+1] < 0).item()
            else:
                fn += 1
        print(f'Logits {fc2}')
        print(f'Truth: {target.iloc[t, unknown]}')
        print('--------------------------------------')
        i += 2
        
    tp /= N
    fn /= N
            
    print('Attack report:')
    print(f'TPR = {tp} | FNR = {fn}')

    tn, fp = 0.0, 0.0
    i = 1
    for t in range(N):
        try: df = data_train.drop(indexes[t], axis=0, inplace=False)
        except: df = data_train
            
        neg = df.shape[0]
        test = torch.tensor(np.array(df.values).astype(np.float32))
        
        with torch.no_grad(): out, probs, fc2 = adv_model(test)
        tn += (torch.sum(fc2[:, i] < 0).item() + torch.sum(fc2[:, i+1] < 0).item())/(2*neg)
        fp += (torch.sum(fc2[:, i] > 0).item() + torch.sum(fc2[:, i+1] > 0).item())/(2*neg)
        #print(f'Class {i}: TNR = {true_negatives/neg} | FPR = {false_positives/neg}')
        i += 2
    tn /= N
    fp /= N
    print(f'TNR = {tn} | FPR = {fp}')
    
    return tp, fn, tn, fp


def WADM_eval(N, model_w, global_model, adv_model, alpha, total_neurons, save_path, mitigate=False):
    print('\n\n============== Changing weights ==============')
    # change main task model weights
    adv_w = adv_model.state_dict()
    layers = [l for l in model_w.keys() if '0' in l.split('.') or '1' in l.split('.')]
    bef = copy.deepcopy(model_w)
    global_model[layers[0]] = adv_w['fc1.weight']
    global_model[layers[1]] = adv_w['fc1.bias']

    if 2*N > 1:
        for i in range(1, 2*N+1):
            global_model[layers[2]][-i,:] = adv_w['fc2.weight'][2*N+1-i,:]# neuron weights
            global_model[layers[3]][-i] = adv_w['fc2.bias'][2*N+1-i]# neuron bias
    else:
        global_model[layers[2]][-1,:] = adv_w['fc2.weight'][0,:]# neuron weights
        global_model[layers[3]][-1] = adv_w['fc2.bias'][0]# neuron bias
    torch.save(global_model, save_path) # save attacked model
             
    print('\n\n============== Malicious neuron detection ==============')
    tpr, fpr, missed, index = malicious_neuron_detection(bef=bef[layers[2]], cur=global_model[layers[2]],
                              neurons=N*2, total=total_neurons, alpha=alpha) # total_neurons = params_dict['linears'][1][1]
    
    
    if not(mitigate): return tpr, fpr, missed, index

    model_w = replace(bef=bef, cur=global_model, layers=layers[2:], index=index)
    if 2*N > 1: # switch malicious by benign neurons
        for i in range(1, 2*N+1):
            adv_w['fc2.weight'][(i-1),:] = model_w[layers[2]][-i,:] # neuron weights
            adv_w['fc2.bias'][(i-1)] = model_w[layers[3]][-i] # neuron bias
    else:
        adv_w['fc2.weight'][0,:] = model_w[layers[2]][-1,:] # neuron weights
        adv_w['fc2.bias'][0] = model_w[layers[3]][-1] # neuron bias
    adv_model.set_parameters(model_w)
    adv_w = adv_model.state_dict()
    return tpr, fpr, missed, index, adv_w # attack performance with these adv_w model weights
    

def realistic_eval(N, data_train, adv_model, global_model, neurons, remove, mode='attack'):
    # 2 options of realistic evaluation of MIA2AIA
    #    1) In case both neurons are activated (inconclusive) choose the one that would corresponds to the most frequent attribute value (in accordance with the theoretical evaluation)
    #    2) In case both neurons are activated (inconclusive) choose the one that yielded a greater gradient
    DEVICE = torch.device('cpu')
    criterion = nn.BCELoss().to(DEVICE)
    
    n_input = global_model['linears.0.weight'].shape[1]
    numneurons = global_model['linears.0.weight'].shape[0]
    secondFC = global_model['linears.1.weight'].shape[0]
    params_dict = {'linears': [(n_input, numneurons), (numneurons, secondFC), (secondFC, 1)],
                   'actvs': ['ReLU', 'ReLU', 'Sigmoid'],
                   'loss': 'BCELoss', 'optimizer': 'Adam'}
    
    model = Model(params_dict)

    # change weights
    layers = [l for l in global_model.keys() if '0' in l.split('.') or '1' in l.split('.')]
    global_model[layers[0]] = adv_model.state_dict()['fc1.weight']
    global_model[layers[1]] = adv_model.state_dict()['fc1.bias']
    
    if mode == 'attack': # assuming single target attacks
        global_model[layers[2]][-1,:] = adv_model.state_dict()['fc2.weight'][1,:]
        global_model[layers[3]][-1] = adv_model.state_dict()['fc2.bias'][1]
        global_model[layers[2]][-2,:] = adv_model.state_dict()['fc2.weight'][2,:]
        global_model[layers[3]][-2] = adv_model.state_dict()['fc2.bias'][2]

    if mode == 'mitigate': # assuming single target attacks
        # WADM mitigation phase
        adv_w = adv_model.state_dict()
        adv_w['fc2.weight'][1,:] = global_model[layers[2]][-1,:]
        adv_w['fc2.bias'][1] = global_model[layers[3]][-1]
        adv_w['fc2.weight'][2,:] = global_model[layers[2]][-2,:]
        adv_w['fc2.bias'][2] = global_model[layers[3]][-2]
        adv_model = adv_model_init(numneurons, global_model=None, n_target=2, n_input=n_input)
        adv_model.load_state_dict(adv_w)

    model.load_state_dict(global_model)
    model.to(DEVICE)
    adv_model.to(DEVICE)
    
    gradients = []
    weights = model.state_dict()['linears.2.weight']
    tuples = torch.tensor(np.array(data_train.iloc[:, :-remove].values).astype(np.float32)).to(DEVICE)
    labels = torch.tensor(data_train.iloc[:, -remove].values, dtype=torch.float32).to(DEVICE)

    i, j = 1, 0
    confs = model(tuples)
    _, _, fc2 = adv_model(tuples)
    fc2 = fc2.detach()
    grad1 = np.sum([BCELoss_prime(conf.item(), y.item())*relu_prime(fc2[k, i])*weights[:, neurons[j]].item() for k, conf, y in zip(range(data_train.shape[0]), confs, labels)])
    grad2 = np.sum([BCELoss_prime(conf.item(), y.item())*relu_prime(fc2[k, i+1])*weights[:, neurons[j+1]].item() for k, conf, y in zip(range(data_train.shape[0]), confs, labels)])
    gradients = [(abs(grad1.item())/data_train.shape[0], abs(grad2.item())/data_train.shape[0])]

    return gradients
    