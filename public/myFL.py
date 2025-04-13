# FEDERATED LEARNING SCHEME - replace flower implementation

import copy
import sys
import numpy as np
import argparse
import torch
import pandas as pd
#from federated_learning_module import FLDataset
from sklearn.model_selection import train_test_split
from centralized_training import train, test, Model
import multiprocessing as mp


class Server:
    def __init__(self, model):
        self.model = model

    def aggregate_gradients(self, client_gradients):
        # Averaging gradients received from clients
        averaged_gradients = np.mean(client_gradients, axis=0)

        # Update the global model with the averaged gradients
        self.model.update_parameters(averaged_gradients)

    def modify_parameters(self):
        # COMPLETE
        pass

    def inference_time(self):
        # COMPLETE
        pass



class Client:
    def __init__(self, ID, train_data, test_data, local_model):
        self.id = ID
        self.train_data = train_data
        self.test_data = test_data
        self.local_model = local_model
        self.copy = copy.deepcopy(self.local_model.state_dict())


    def train(self, hyperparameters):
        # Train the local model on client's data
        new_model, results = train(self.train_data, self.local_model, **hyperparameters)
        weight = len(self.train_data.dataset)
        #weight = 1
        
        gradients = copy.deepcopy(new_model.state_dict())
        for k, v in self.copy.items():
            gradients[k] -= v
            gradients[k] *= weight

        self.local_model = new_model
        self.copy = copy.deepcopy(self.local_model.state_dict())

        for metric in results.keys():
            results[metric] *= weight
        
        return results, gradients


    def test(self, hyperparameters):
        
        results = test(self.test_data, self.local_model, **hyperparameters)
        #weight = 1
        weight = len(self.test_data.dataset)
        for metric in results.keys():
            results[metric] *= weight
            
        return results

    
class Simulation:
    def __init__(self, seed, strategy, n_clients, clients_training_data, clients_test_data, params_dict, hyperparameters, save_path='COPMODE_fl.pth'):
        self.seed = seed
        self.strategy = strategy
        self.n_clients = n_clients
        self.clients_training_data = clients_training_data # use dataset class - list of dataloaders
        self.clients_test_data = clients_test_data
        self.params_dict = params_dict
        self.hyperparameters = hyperparameters # dict containing lr, epochs, criterion, optimizer, metrics, verbose (to send for train and test), DEVICE
        #self.fraction_clients = fraction_clients
        self.save_path = save_path
        

    def start(self, DEVICE, rounds, global_model=None):
        # reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        #rng = np.random.default_rng(seed=self.seed)
        
        if global_model == None:
            global_model = Model(self.params_dict)

        print('Starting FL training ...')
            
        # Federated learning iterations
        for r in range(rounds):        
            train_results = []
            test_results = []
            gradients = []
            n_train = 0
            n_test = 0
            clients = []

            global_model.to(DEVICE)
            params_copy = copy.deepcopy(global_model.state_dict())
            for k, v in global_model.state_dict().items():
                print(torch.sum(v))
            
            print(f'- Round {r}')
            #selection = list(np.random.randint(low=0, high=93, size=93))
            
            for c in range(self.n_clients):
                #if c in selection:
                clients.append(Client(c, self.clients_training_data[c], self.clients_test_data[c], copy.deepcopy(global_model)))
            
            # Clients train local models and send gradients to the server - PARALELIZAR LOOP
            for c, client in enumerate(clients):
                #print(f'- - Client {c} ({len(client.train_data.dataset)})')
                
                result, gradient = client.train(self.hyperparameters)
                train_results.append(result)
                
                result = client.test(self.hyperparameters)
                test_results.append(result)
                
                gradients.append(gradient)
                n_train += len(client.train_data.dataset)
                n_test += len(client.test_data.dataset)

            # Server aggregates gradients and updates the global model
            #aggr_gradients = aggregation(gradients, n_train) # 93
            #global_model = update(Model(self.params_dict), params_copy, aggr_gradients)

            #n = np.sum(weights)
            global_model_params = {k: torch.zeros_like(v) for k,v in global_model.state_dict().items()}
            for k in global_model_params.keys():
                for i, client in enumerate(clients):
                    global_model_params[k] += client.local_model.state_dict()[k]*len(client.train_data.dataset)/n_train

            global_model.load_state_dict(global_model_params, strict=True)
            global_model.to(DEVICE)
            
            # Report
            agg_results = aggregate_results(train_results, n_train) # 93
            print('Training report')
            report(agg_results)
            
            agg_results = aggregate_results(test_results, n_test) # 93
            print('Testing report')
            report(agg_results)
            
        torch.save(global_model.state_dict(), self.save_path)
        #for k, v in global_model.state_dict().items():
        #    print(torch.sum(v))


    def parallel_start(self, DEVICE, rounds, model=None):
        # reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        #rng = np.random.default_rng(seed=self.seed)
        
        if model == None:
            global_model = Model(self.params_dict)

        print('Starting FL training ...')
            
        # Federated learning iterations
        for r in range(rounds):        
            #train_results = []
            #test_results = []
            #gradients = []
            n_train = 0
            n_test = 0
            clients = []

            global_model.to(DEVICE)
            params_copy = copy.deepcopy(global_model.state_dict())
            
            print(f'- Round {r}')
            
            for c in range(min(8, self.n_clients)):
                clients.append(Client(c, self.clients_training_data[c], self.clients_test_data[c], copy.deepcopy(global_model)))
                n_train += len(self.clients_training_data[c].dataset)
                n_test += len(self.clients_test_data[c].dataset)

            def train_client(client):
                return client.train(self.hyperparameters)

            def test_client(client):
                return client.test(self.hyperparameters)
            
            # Clients train local models and send gradients to the server
            # Training
            pool = mp.Pool(processes=4)
            train_results_tuples = pool.map_async(train_client, clients)
            pool.close()
            pool.join()

            # Test
            pool = mp.Pool(processes=4)
            test_results = pool.map_async(test_client, clients)
            pool.close()
            pool.join()

            train_results_tuples = train_results_tuples.get()
            test_results = test_results.get()
            
            gradients = [m for r,m in train_results_tuples]
            train_results = [r for r,m in train_results_tuples]

            # Server aggregates gradients and updates the global model
            aggr_gradients = aggregation(gradients, n_train)
            global_model = update(Model(self.params_dict), params_copy, aggr_gradients)
            
            # Report
            agg_results = aggregate_results(train_results, n_train)
            print('Training report')
            report(agg_results)
            
            agg_results = aggregate_results(test_results, n_test)
            print('Testing report')
            report(agg_results)

            
        torch.save(global_model.state_dict(), self.save_path)


        

def aggregation(gradients, n):
    # aggregate gradients according to weights
    #n = np.sum(weights)
    #aggregated_gradient = {k: gradients[0][k]*(weights[0]/n) for k in gradients[0].keys()}
    aggregated_gradient = {k: torch.zeros_like(gradients[0][k]) for k in gradients[0].keys()}
    
    for k in aggregated_gradient.keys():
        for i in range(len(gradients)):
            #aggregated_gradient[k] += gradients[i][k]*(weights[i]/n)
            aggregated_gradient[k] += gradients[i][k]/n
    
    return aggregated_gradient


def update(model, params_copy, gradients):
    #update = {k: torch.zeros_like(gradients[k]) for k in gradients.keys()}
    
    #for k in params_copy.keys():
    #    for i in range(len(weights)):
    #        update[k] += params_copy[k]*(weights[i]/np.sum(weights))
    #    print(torch.sum(update[k]) - torch.sum(params_copy[k]))
    
    for k in params_copy.keys():
        params_copy[k] += gradients[k]

    model.load_state_dict(params_copy, strict=True)
    return model


def aggregate_results(results, n):
    out = dict(zip(list(results[0].keys()), np.zeros(len(results[0]))))
    #n = np.sum(weights)
    
    for metric in out.keys():
        for i, r in enumerate(results):
            #out[metric] += r[metric]*(weights[i]/n)
            out[metric] += r[metric]/n

    return out


def report(results):
    for metric in results.keys():
        name = metric.split('_')[:-1][0]
        str_metric = "{:.8f}".format(results[metric])
        k = 21-len(name)
        print(f'{" ":>8}{name}{str_metric:>{k}}\n')


def parse_list_int(value):
    try:
        # Split the input string by commas and convert each part to an integer
        linears = [int(x) for x in value.split(',')]
        return linears
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid value for linears. Please provide a comma-separated list of integers.")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FL system')
    parser.add_argument('-s', '--seed', type=int, default=4)
    parser.add_argument('-st', '--strategy', type=str, default='FedAVG')
    parser.add_argument('-c', '--n_clients', type=int, default=10)
    parser.add_argument('-sh', '--shadow', type=int, default=5009)
    parser.add_argument('-p', '--train_size', type=float, default=0.8)
    parser.add_argument('-lr', '--lr', type=float, default=1e-4)
    parser.add_argument('-e', '--epochs', type=int, default=25)
    parser.add_argument('-l', '--linears', type=parse_list_int, default=[100], help='Split ints by commas')
    parser.add_argument('-a', '--actvs', type=str, default='ReLU')
    parser.add_argument('-o', '--outfunc', type=str, default='Sigmoid')
    #parser.add_argument('-o', '--save_path', type=str, required=True)
    
    args = parser.parse_args()
    DEVICE = torch.device('cpu')
    # missing example
