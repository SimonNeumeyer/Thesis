import numpy
import torch
import torch.nn as nn
from util import Constants
from myLogging import alpha_gradient_logging
from multiprocessing import reduction
from collections import OrderedDict

class DiffNN(nn.ModuleList):
    
    def __init__(self, modules, optimization_settings):
        super(DiffNN, self).__init__(modules)
        self.alphas = self.init_alphas(modules)
        self.set_alpha_update(False)
        self.name = "test" #TODO
        self.optimization_settings = optimization_settings
        
    def get_name(self):
        return self.name
    
    def set_optimization_settings(self, optimization_settings):
        self.optimization_settings = optimization_settings
        
    def set_alpha_update(self, alpha_update):
        """ Freeze or unfreeze weights respectively alphas """
        self.alpha_update = alpha_update
        if alpha_update:
            self.alphas.requires_grad = True
            self.set_weight_update(False)
        else:
            self.alphas.requires_grad = False
            self.set_weight_update(True)
            
    def get_alphas(self):
        return self.alphas.data.clone().detach()
            
    def get_weight_parameters(self):
        return [parameter[1] for parameter in self.named_parameters() if "alphas" not in parameter[0]]
            
    def set_weight_update(self, weight_update):
        for parameter in self.get_weight_parameters():
            if weight_update:
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False
        
    def calculate_alphas(self, alpha_sampling):
        """ alphas are either sampled or smooth """
        alphas = nn.functional.softmax(self.alphas)
        if alpha_sampling:
            alphas = nn.functional.gumbel_softmax(self.alphas, hard=True, tau=1)
        return alphas
        
    def init_alphas(self, modules):
        alphas = nn.Parameter(torch.ones(len(modules)), requires_grad=True)
        #alphas.register_hook(alpha_gradient_logging)
        return alphas
    
    def alpha_gradient_active(self):
        return self.alpha_update
        
    def reduce(self, alphas, tensors, reduce, normalize):
        #print(torch.stack(tensors, dim=1).shape)
        if reduce == Constants.REDUCE_FUNC_SUM:
            #reduced = nn.functional.linear(alphas, torch.stack(tensors, dim=2))
            reduced = torch.matmul(alphas, torch.stack(tensors, dim=1))
        else:
            raise NotImplementedError("no other reduction function than sum")
        if normalize:
            return reduced / torch.linalg.norm(reduced)
        else:
            return reduced
        
    def forward(self, x):
        if self.optimization_settings.alpha_update and not self.alpha_update:
            self.set_alpha_update(True)
        if not self.optimization_settings.alpha_update and self.alpha_update:
            self.set_alpha_update(False)
        alphas = self.calculate_alphas(self.optimization_settings.alpha_sampling)
        return self.reduce(alphas, [module(x, self.optimization_settings) for module in self], reduce=self.optimization_settings.diffNN_reduce, normalize=self.optimization_settings.diffNN_normalize)
        
        
class GraphNN(nn.Module):
    
    def __init__(self, graph, width):
        super(GraphNN, self).__init__()
        self.graph = graph
        self.width = width
        self.edgeNN = "EdgeNN"
        self.initModel()
        
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
        
    def reduce(self, tensors, reduce, normalize):
        if reduce == Constants.REDUCE_FUNC_SUM:
            reduced = torch.sum(torch.stack(tensors), dim=0)
        else:
            raise NotImplementedError("no other reduction function than sum")
        if normalize:
            return reduced / torch.linalg.norm(reduced)
        else:
            return reduced
        
    def forward(self, x, optimization_settings):
        outputs = {self.graph.input_node : x}
        for v in self.graph.ordered_nodes(except_input_node = True):
            inputs = [self.graph.get_edge_attribute(p, v)[self.edgeNN](outputs[p]) for p in self.graph.get_predecessors(v)]
            outputs[v] = self.reduce(inputs, reduce=optimization_settings.graphNN_reduce, normalize=optimization_settings.graphNN_normalize)
        return outputs[self.graph.output_node]