from diffNN import *
from visualization import *
from util import *
from data import *
import graph
import torch
import torch.nn as nn
from torch.optim import SGD

class Trainer():
    
    def __init__(self, number_nodes=3, number_graphs=3, visualize=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initiate_data()
        self.visualization = Visualization(visualize)
        self.optimization_settings = OptimizationSettings()
        graphs = graph.GraphGenerator(number_nodes).get_random_subset(number_graphs)
        if self.optimization_settings.shared_weights:
            graphNNs = GraphNN.generate_graphNNs_shared_weights(graphs, self.number_features)
        else:
            graphNNs = [GraphNN(graph, self.number_features) for graph in graphs]
        diffNN = DiffNN(graphNNs, self.optimization_settings)
        self.visualization.plot_graphs([g.get_networkx() for g in graphs])
        self.visualization.register_diffNN(diffNN)
        self.model = nn.Sequential(diffNN, nn.Linear(self.number_features, self.number_classes), nn.Softmax(dim=-1))
        self.model.to(self.device)
        
        self.loss_function, self.optimizer = self.backward_stuff(self.model)
        
    def initiate_data(self):
        #dataset = MyDataset(overwrite=True)
        dataset = MyMNIST()
        self.loader_train, self.loader_test = dataset.get_train_test_loader()
        self.number_classes = dataset.number_classes
        self.number_features = dataset.number_features
        
        
    def backward_stuff(self, model):
        return (nn.CrossEntropyLoss(), SGD(model.parameters(), lr=1e-1, momentum=0.8))
    
    def run(self):
        for i in range(5):
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
        for i, o in self.loader_train:
            i, o = i.to(self.device), o.to(self.device)
            loss = self.loss(i, o)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.visualization.batch(copy_tensor(loss))
        self.visualization.epoch(copy_tensor(self.evaluate()))
        for m in self.model:
            if isinstance(m, nn.Linear):
                self.visualization.plot_weight_matrix(copy_tensor(m.weight), "Last linear")
    
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
        return loss
    
    def update_optimization_settings(self, **kwargs):
        self.optimization_settings.update(**kwargs)
        for m in self.model:
            if isinstance(m, DiffNN):
                m.set_optimization_settings(self.optimization_settings)
                
    def loss(self, i, o):
        if isinstance(self.loss_function, nn.CrossEntropyLoss):
            _, o = o.max(dim=-1) #get indices from one-hot encoding
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
        self.graphNN_normalize = True
        self.diffNN_normalize = False
        self.shared_weights = False
        
    def update(self, **kwargs):
        for key in kwargs:
            assert key in self.__dict__, f"Setting '{key}' not available"
            self.__dict__[key] = kwargs[key]
    
        
if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()