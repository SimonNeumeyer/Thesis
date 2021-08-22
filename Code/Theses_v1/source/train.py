from visualization import *
from util import *
from data import *
from model import *
from settings import Settings
import json
import re
import sys
import graph
import torch
import torch.nn as nn
from torch.optim import SGD, Adam

class Trainer():
    
    def __init__(self, **kwargs):
        self.settings = Settings(**kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Cuda: {torch.cuda.is_available()}")
        features, classes = self.initiate_data(self.settings["optimization"]["batchSize"])
        self.settings.update(**{"model": {"classes": classes, "features": features}})
        self.visualization = Visualization(self.settings["org"]["visualization"], json.dumps(self.settings.settings))
        self.model = Model0(self.settings["model"])
        self.model.to(self.device)
        self.visualization.plot_model(self.model, self.dataset.get_sample()[0].to(self.device))
        self.loss_function, self.optimizer = self.backward_stuff(self.model, self.settings["optimization"]["optimizer"])
        self.visualization.close()
        
    def initiate_data(self, batchSize):
        #self.dataset = MyDataset(overwrite=False)
        self.dataset = MyMNIST()
        self.loader_train, self.loader_test = self.dataset.get_train_test_loader(batchSize)
        return (self.dataset.number_features, self.dataset.number_classes)
        
        
    def backward_stuff(self, model, optimizer):
        if optimizer == Constants.OPTIMIZER_ADAM:
            return (nn.CrossEntropyLoss(), Adam(model.parameters(), lr=1e-2))
        else:
            return (nn.CrossEntropyLoss(), SGD(model.parameters(), lr=1e-2, momentum=0.8))
    
    def run(self):
        for i in range(self.settings["optimization"]["epochs"]):
            print(f"epoch {i+1}")
            self.train_epoch(i)
            self.evaluate(i)
    
    def train_epoch(self, epoch):
        self.model.train()
        counter = 0
        for i, o in self.loader_train:
            counter += 1
            self.alpha_update(epoch, counter)
            i, o = i.to(self.device), o.to(self.device)
            loss = self.loss(i, o)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.visualization.training_loss(epoch, counter, copy_tensor(loss))
        #self.visualization.plot_weight_matrix(copy_tensor(self.model.LastLinear.weight), "Last linear")
    
    def evaluate(self, epoch):
        self.model.eval()
        with torch.no_grad():
            loss = 0
            accuracy = 0
            batchCount = 0
            for i, o in self.loader_test:
                i, o = i.to(self.device), o.to(self.device)
                batchCount +=1
                loss = ((batchCount - 1) * loss + self.loss(i, o)) / batchCount
                accuracy = ((batchCount - 1) * accuracy + self.accuracy(i, o)) / batchCount
        self.model.train()
        self.visualization.evaluation_loss(epoch, evaluation_loss=copy_tensor(loss), accuracy=copy_tensor(accuracy))
        return loss

    def alpha_update(self, epoch, counter):
        if epoch % 2 == 0:
            alphaUpdate = True
        else:
            alphaUpdate = False
        self.settings.update({"optimization": {"alphaUpdate": alphaUpdate}})
                
    def loss(self, i, o):
        if isinstance(self.loss_function, nn.CrossEntropyLoss):
            _, o = o.max(dim=-1) #get indices from one-hot encoding
        return self.loss_function(self.model(i), o)

    def accuracy (self, i, o):
        return torch.true_divide((self.model(i).max(dim=-1)[1] == o.max(dim=-1)[1]).sum(), i.size()[0])
        
    def close(self):
        self.visualization.close()
        

def parse_args(list):
    args = {}
    for item in list:
        assert len(item.split("=")) == 2, f"args syntax unfamiliar: {item}"
        try:
            args[item.split("=")[0]] = int(item.split("=")[1])
        except:
            try:
                args[item.split("=")[0]] = bool(item.split("=")[1])
            except:
                args[item.split("=")[0]] = item.split("=")[1]
    return args
        
if __name__ == "__main__":
    #args = parse_args(sys.argv[1:])
    trainer = Trainer(**{})
    trainer.run()
