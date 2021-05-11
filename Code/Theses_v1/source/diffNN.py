import numpy
import torch
import torch.nn as nn
import util
from myLogging import alpha_gradient_logging

class DiffNN(nn.ModuleList):
    
    def __init__(self, modules = None):
        super(DiffNN, self).__init__(modules)
        self.alphas = self.init_alphas(modules)
        self.set_alpha_update(False)
        
    def set_alpha_update(self, alpha_update):
        """ Freeze or unfreeze weights respectively alphas """
        self.alpha_update = alpha_update
        if alpha_update:
            self.alphas.requires_grad = True
            self.set_weight_update(False)
        else:
            self.alphas.requires_grad = False
            self.set_weight_update(True)
            
    def get_weight_parameters(self):
        return [parameter[1] for parameter in self.named_parameters() if 'alphas' not in parameter[0]]
            
    def set_weight_update(self, weight_update):
        for parameter in self.get_weight_parameters():
            if weight_update:
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False
        
    def get_alphas(self, alpha_sampling):
        """ alphas are either sampled or smooth """
        alphas = nn.functional.softmax(self.alphas)
        if alpha_sampling:
            alphas = nn.functional.gumbel_softmax(self.alphas, hard=True, tau=1)
        return alphas
        
    def init_alphas(self, modules):
        number_modules = 0
        if modules:
            number_modules = len(modules)
        alphas = nn.Parameter(torch.ones(number_modules), requires_grad=True)
        alphas.register_hook(alpha_gradient_logging)
        return alphas
        
    def forward(self, x, reduction = util.Reduction.SUM, alpha_update = False, alpha_sampling = False):
        if alpha_update and not self.alpha_update:
            self.set_alpha_update(True)
        if not alpha_update and self.alpha_update:
            self.set_alpha_update(False)
        alphas = self.get_alphas(alpha_sampling)
        if reduction == util.Reduction.SUM:
            return nn.functional.linear(alphas, torch.stack([module(x) for module in self], dim=1))
        
        
class Graph_nn(nn.Module):
    
    def __init__(self, graph, width):
        super(Graph_nn, self).__init__()
        self.model = nn.Linear(width, width)
        
    def forward(self, x):
        return self.model(x)