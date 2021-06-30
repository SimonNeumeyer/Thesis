from diffNN import *
from visualization import *
from util import *
from data import *
import json
import re
import sys
import graph
import torch
import torch.nn as nn
from torch.optim import SGD, Adam

class Trainer():
    
    def __init__(self, number_nodes=3, number_graphs=2, number_features_graphNN=59, epochs=3, visualize=False, optimizer='adam'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.number_features_graphNN = number_features_graphNN
        print(f"Cuda: {torch.cuda.is_available()}")
        self.initiate_data()
        args = re.sub('[:\\s"\\{\\}]', '_', json.dumps({'number_nodes': number_nodes, 'number_graphs': number_graphs,
            'epochs': epochs, 'optimizer': optimizer, 'number_features_graphNN': number_features_graphNN}))
        self.visualization = Visualization(args, visualize)
        self.optimization_settings = OptimizationSettings()
        graphs = graph.GraphGenerator(number_nodes).get_random_subset(number_graphs)
        if self.optimization_settings.shared_weights:
            graphNNs = GraphNN.generate_graphNNs_shared_weights(graphs, self.number_features_graphNN)
        else:
            graphNNs = [GraphNN(graph, self.number_features_graphNN) for graph in graphs]
        diffNN = DiffNN(graphNNs, self.optimization_settings)
        self.visualization.plot_graphs([g.get_networkx() for g in graphs])
        self.visualization.register_diffNN(diffNN)
        self.model = nn.Sequential()
        self.model.add_module("FirstLinear", nn.Linear(self.number_features_in, self.number_features_graphNN))
        self.model.add_module("DiffNN", diffNN)
        self.model.add_module("LastLinear", nn.Linear(self.number_features_graphNN, self.number_classes))
        self.model.add_module("Softmax", nn.Softmax(dim=-1))
        self.model.to(self.device)
        self.visualization.plot_model(self.model, self.dataset.get_sample()[0])
        self.loss_function, self.optimizer = self.backward_stuff(self.model, optimizer)
        
    def initiate_data(self):
        #dataset = MyDataset(overwrite=True)
        self.dataset = MyMNIST()
        self.loader_train, self.loader_test = self.dataset.get_train_test_loader()
        self.number_classes = self.dataset.number_classes
        self.number_features_in = self.dataset.number_features
        
        
    def backward_stuff(self, model, optimizer):
        if optimizer == 'adam':
            return (nn.CrossEntropyLoss(), Adam(model.parameters(), lr=1e-2))
        else:
            return (nn.CrossEntropyLoss(), SGD(model.parameters(), lr=1e-2, momentum=0.8))
    
    def run(self):
        for i in range(self.epochs):
            print(f"epoch {i+1}")
            self.update_optimization_settings(alpha_update=False)
            for j in range(1):
                self.train_epoch()
                self.evaluate()
            self.update_optimization_settings(alpha_update=True)
            for j in range(1):
                self.train_epoch()
                self.evaluate()
    
    def train_epoch(self):
        self.model.train()
        counter = 0
        for i, o in self.loader_train:
            counter += 1
            i, o = i.to(self.device), o.to(self.device)
            loss = self.loss(i, o)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.visualization.batch(copy_tensor(loss))
        self.visualization.plot_weight_matrix(copy_tensor(self.model.LastLinear.weight), "Last linear")
    
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            loss = 0
            batch_count = 0
            for i, o in self.loader_test:
                i, o = i.to(self.device), o.to(self.device)
                batch_count +=1
                loss = ((batch_count - 1) * loss + self.loss(i, o)) / batch_count
        self.model.train()
        self.visualization.epoch(copy_tensor(loss))
        return loss
    
    def update_optimization_settings(self, **kwargs):
        self.optimization_settings.update(**kwargs)
        for m in self.model:
            if isinstance(m, DiffNN):
                m.set_optimization_settings(self.optimization_settings)
                
    def loss(self, i, o):
        if isinstance(self.loss_function, nn.CrossEntropyLoss):
            _, o = o.max(dim=-1) #get indices from one-hot encoding
        #if counter % 100 == 0:
            #print(self.model(i))
        return self.loss_function(self.model(i), o)
        
    def close(self):
        self.visualization.close()
        
        
class OptimizationSettings():
    
    def __init__(self, **kwargs):
        self._init_default()
        self.update(**kwargs)
        
    def _init_default(self):
        self.alpha_update = True
        self.alpha_sampling = False
        self.diffNN_reduce = Constants.REDUCE_FUNC_SUM
        self.graphNN_reduce = Constants.REDUCE_FUNC_SUM
        self.graphNN_normalize = False
        self.diffNN_normalize = False
        self.shared_weights = False
        
    def update(self, **kwargs):
        for key in kwargs:
            assert key in self.__dict__, f"Setting '{key}' not available"
            self.__dict__[key] = kwargs[key]
    
def parse_args(list):
    args = {}
    for item in list:
        if len(item.split("=")) > 1:
            try:
                args[item.split("=")[0]] = int(item.split("=")[1])
            except:
                args[item.split("=")[0]] = item.split("=")[1]
        else:
            args[item] = True
    return args
        
if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    trainer = Trainer(**args)
    trainer.run()
