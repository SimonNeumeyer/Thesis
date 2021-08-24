import torch
import torch.nn as nn
from util import Constants
from graph import GraphGenerator
from collections import OrderedDict

class DiffNN(nn.ModuleList):
    
    def __init__(self, name, modules, settings):
        super(DiffNN, self).__init__(modules)
        self.name = name
        self.settings = settings
        self.alphas = self.init_alphas(modules)
        self.alpha_update = True
        self.set_alpha_update(False)
        
    def get_name(self):
        return self.name
        
    def set_alpha_update(self, alpha_update):
        """ Freeze or unfreeze weights respectively alphas """
        if self.alpha_update != alpha_update:
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
        if alpha_sampling:
            return nn.functional.gumbel_softmax(self.alphas, hard=True, tau=1)
        else:
            return nn.functional.softmax(self.alphas, dim=0)
        
    def init_alphas(self, modules):
        if self.settings["darts"]["randomInit"]:
            return nn.Parameter(torch.rand(len(modules)), requires_grad=True)
        else:
            return nn.Parameter(torch.ones(len(modules)), requires_grad=True)
        
    def reduce(self, alphas, tensors, reduce, normalize):
        if reduce == Constants.REDUCE_FUNC_SUM:
            reduced = torch.matmul(alphas, torch.stack(tensors, dim=1))
        else:
            raise NotImplementedError("no other reduction function than sum")
        if normalize:
            return reduced / torch.linalg.norm(reduced)
        else:
            return reduced
        
    def forward(self, x):
        #if self.settings["darts"]["active"] and not self.alpha_update:
        #    self.set_alpha_update(True)
        #if not self.settings["darts"]["active"] and self.alpha_update:
        #    self.set_alpha_update(False)
        alphas = self.calculate_alphas(self.settings["darts"]["sampling"])
        return self.reduce(alphas, [module(x) for module in self], reduce=self.settings["darts"]["reduce"], normalize=self.settings["darts"]["normalize"])
        
        
class GraphNN(nn.Module):
    
    def __init__(self, graph, settings, shared_edgeNNs=None):
        super(GraphNN, self).__init__()
        self.settings = settings
        self.graph = graph
        self.width = settings["features"]
        self.edgeNNs = shared_edgeNNs
        self.initModel()
        
    @classmethod
    def generate_graphNNs_shared_weights(cls, graphs, width):
        assert False, "method call for shared weights not expected"
        dense_graph = GraphGenerator.dense_graph(graphs[0].number_nodes_without_input_output())
        dense_graphNN = GraphNN(graph=dense_graph, width=width)
        graphNNs = []
        for g in graphs:
            graphNNs.append(GraphNN(g, shared_edgeNNs=dense_graphNN.get_edgeNNs()))
        return graphNNs

    def get_edgeNNs (self):
        return self.edgeNNs

    def create_edgeNNs(self, graph, width):
        self.edgeNNs = nn.ModuleDict()
        for edge in graph.edges():
            if not graph.input_output_edge(*edge):
                self.edgeNNs[self.stringify_edge(edge)] = nn.Sequential(nn.Linear(width, width), nn.ReLU())

    def stringify_edge(self, edge):
        return f"edge_{edge[0]}_to_{edge[1]}"
        
    def initModel(self):
        if self.edgeNNs is None:
            self.create_edgeNNs(self.graph, self.width)
        for edge in self.graph.edges():
            if self.graph.input_output_edge(*edge):
                self.edgeNNs[self.stringify_edge(edge)] = nn.Identity()
        
    def reduce(self, tensors, reduce, normalize):
        if reduce == Constants.REDUCE_FUNC_SUM:
            reduced = torch.sum(torch.stack(tensors), dim=0)
        else:
            raise NotImplementedError("no other reduction function than sum")
        if normalize:
            return reduced / torch.linalg.norm(reduced)
        else:
            return reduced
        
    def forward(self, x):
        outputs = {self.graph.input_node : x}
        for v in self.graph.ordered_nodes(except_input_node = True):
            inputs = [self.edgeNNs[self.stringify_edge((p, v))](outputs[p]) for p in self.graph.get_predecessors(v)]
            outputs[v] = self.reduce(inputs, reduce=self.settings["reduce"], normalize=self.settings["normalize"])
        return outputs[self.graph.output_node]
