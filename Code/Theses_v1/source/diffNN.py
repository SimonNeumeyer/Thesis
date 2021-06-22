import torch
import torch.nn as nn
from util import Constants
from graph import GraphGenerator
#from myLogging import alpha_gradient_logging
from collections import OrderedDict
from main import OptimizationSettings #MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM

class DiffNN(nn.ModuleList):
    
    def __init__(self, modules, optimization_settings):
        super(DiffNN, self).__init__(modules)
        self.alphas = self.init_alphas(modules)
        self.set_alpha_update(False)
        self.name = "OnlyDiffNN" #TODO
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
        alphas = nn.functional.softmax(self.alphas, dim=0)
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
    
    def __init__(self, graph, width=None, shared_edgeNNs=None):
        super(GraphNN, self).__init__()
        self.graph = graph
        self.width = width
        self.edgeNN = "EdgeNN"
        self.initModel(shared_edgeNNs)
        
    @classmethod
    def generate_graphNNs_shared_weights(cls, graphs, width):
        graphNNs = []
        shared_edgeNNs = cls.generate_shared_weights(graphs[0], width)
        for g in graphs:
            graphNNs.append(GraphNN(g, shared_edgeNNs=shared_edgeNNs))
        return graphNNs
            
    @classmethod
    def generate_shared_weights(cls, graph, width):
        graph = GraphGenerator.get_maximal_phenotype(graph.number_nodes_without_input_output())
        return cls.create_edgeNNs(graph, width)
    
    @classmethod
    def create_edgeNNs(cls, graph, width):
        edgeNNs = {}
        for edge in graph.edges():
            if graph.input_output_edge(*edge):
                edgeNN = nn.Identity() 
            else:
                edgeNN = nn.Sequential(nn.Linear(width, width), nn.ReLU())
            edgeNNs[str(edge)] = edgeNN
        return edgeNNs
        
    def initModel(self, edgeNNs):
        if edgeNNs is None:
            edgeNNs = self.create_edgeNNs(self.graph, self.width)
        for edge in self.graph.edges():
            if self.graph.input_output_edge(*edge):
                edgeNN = nn.Identity() 
            else:
                edgeNN = edgeNNs[str(edge)]
            self.graph.set_edge_attribute(*edge, self.edgeNN, edgeNN) # self.edgeNN is string constant
        self.edgeNNs = nn.Sequential(OrderedDict(edgeNNs)) # register model
        
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
    #def forward(self, x):
        #optimization_settings = OptimizationSettings()
        outputs = {self.graph.input_node : x}
        for v in self.graph.ordered_nodes(except_input_node = True):
            inputs = [self.graph.get_edge_attribute(p, v)[self.edgeNN](outputs[p]) for p in self.graph.get_predecessors(v)]
            outputs[v] = self.reduce(inputs, reduce=optimization_settings.graphNN_reduce, normalize=optimization_settings.graphNN_normalize)
        #print([(i, torch.sum(v)) for (i,v) in outputs.items()])
        return outputs[self.graph.output_node]