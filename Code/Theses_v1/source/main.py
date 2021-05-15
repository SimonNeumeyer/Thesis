import diffNN
import util
import graph
import torch
import torch.nn as nn
from torch.optim import SGD

def visual(model):
    for (n, p) in model.named_parameters():
        print(n)
        print(p)
        print('\n')
        
def backward_stuff(model):
    return (nn.MSELoss(), SGD(model.parameters(), lr=1e-2, momentum=0.8))
        
if __name__ == "__main__":
    parameter = {"reduction": util.Reduction.SUM, "alpha_update": True, "alpha_sampling": True}
    number_nodes = 3
    features_per_node = 10
    graphs = graph.GraphGenerator(number_nodes)[2]
    diffNN = diffNN.DiffNN([diffNN.Graph_nn(graph, features_per_node) for graph in graphs])
    x = torch.ones(features_per_node)
    o = torch.ones(features_per_node)
    loss_function, optimizer = backward_stuff(diffNN)
    for i in range(2):
        loss = loss_function(diffNN(x, **parameter), o)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        visual(diffNN)