import numpy
import torch
import torch.nn as nn
import util
from myLogging import alpha_gradient_logging
from multiprocessing import reduction
from collections import OrderedDict

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
        
        
class GraphNN(nn.Module):
    
    def __init__(self, graph, width):
        super(GraphNN, self).__init__()
        self.graph = graph
        self.width = width
        self.edgeNN = "EdgeNN"
        self.initModel()
        print("parameters:")
        for p in self.named_parameters():
            print(p)
        
    def initModel(self):
        edgeNNs = []
        for edge in self.graph.edges():
            if self.graph.input_output_edge(*edge):
                edgeNN = nn.Identity() 
            else:
                edgeNN = nn.Linear(self.width, self.width)
            edgeNNs.append((str(edge), edgeNN))
            self.graph.set_edge_attribute(*edge, self.edgeNN, edgeNN)
        self.edgeNNs = nn.Sequential(OrderedDict(edgeNNs))
        
    def reduce(self, tensors):
        reduced = torch.sum(torch.stack(tensors), dim=0)
        return reduced / torch.linalg.norm(reduced)
        
    def forward(self, x):
        outputs = {self.graph.input_node : x}
        for v in self.graph.ordered_nodes(except_input_node = True):
            outputs[v] = self.reduce([self.graph.get_edge_attribute(p, v)[self.edgeNN](outputs[p]) for p in self.graph.get_predecessors(v)])
        return outputs[self.graph.output_node]