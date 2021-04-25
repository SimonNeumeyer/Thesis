import numpy
import torch
import torch.nn as nn

class DiffNN(nn.ModuleList):
    
    def __init__(self, modules = None):
        super(DiffNN, self).__init__(modules)
        self.alphas = self.init_alphas(modules)
        self.combination_function = "sum"
        
    def init_alphas(self, modules):
        number_modules = 0
        if modules:
            number_modules = len(modules)
        return nn.Parameter(torch.ones(number_modules), requires_grad=True)
        
    def forward(self, x):
        if self.combination_function == "sum":
            return nn.functional.linear(self.alphas, torch.stack([module(x) for module in self], dim=1))