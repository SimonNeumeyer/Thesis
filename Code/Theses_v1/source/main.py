from diffNN import *
from visualization import *
from util import *
from data import *
import graph
import torch
import torch.nn as nn
from torch.optim import SGD

class Trainer():
    
    def __init__(self):
        self.initiate_data()
        self.visualization = Visualization()
        self.optimization_settings = OptimizationSettings()
        graphs = graph.GraphGenerator(number_nodes=3).get_random_subset(6)
        diffNN = DiffNN([GraphNN(graph, self.number_features) for graph in graphs], self.optimization_settings)
        self.visualization.plot_graphs([g.get_networkx() for g in graphs])
        self.visualization.register_diffNN(diffNN)
        self.model = nn.Sequential(diffNN, nn.Linear(self.number_features, self.number_classes), nn.Softmax())
        
        self.loss_function, self.optimizer = self.backward_stuff(self.model)
        #self.input = torch.ones(features_per_node)
        #self.output = torch.ones(features_per_node)
        
    def initiate_data(self):
        dataset = MyDataset()
        #dataset = MyMNIST(path_load=path_load)
        self.loader_train, self.loader_test = dataset.get_train_test_loader()
        self.number_classes = dataset.number_classes
        self.number_features = dataset.number_features
        
        
    def backward_stuff(self, model):
        return (nn.MSELoss(), SGD(model.parameters(), lr=1e-2, momentum=0.8))
    
    def run(self):
        for i in range(1):
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
        for i, o in self.loader_train:
            loss = self.loss_function(self.model(i), o)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.visualization.batch(loss)
        self.visualization.epoch(self.evaluate())
    
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            loss = 0
            batch_count = 0
            for i, o in self.loader_test:
                batch_count +=1
                loss = ((batch_count - 1) * loss + self.loss_function(self.model(i), o)) / batch_count
        self.model.train()
        return loss
    
    def update_optimization_settings(self, **kwargs):
        self.optimization_settings.update(**kwargs)
        for m in self.model:
            if isinstance(m, DiffNN):
                m.set_optimization_settings(self.optimization_settings)
        
    def close(self):
        self.visualization.close()
        
    
        
if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()