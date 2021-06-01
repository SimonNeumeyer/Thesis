import torch 
import torch.nn.functional as functional
import torch.nn as nn
from myLogging import alpha_gradient_logging
import diffNN
import graph

if False:
    print(hash((2,3)))
    print(hash((3,3)))
    print(hash((2,3)))

#a = torch.ones(5)
# a = torch.tensor([-5,-1,2,3,4,5], dtype=torch.float32)
# a.requires_grad = True
# a.register_hook(alpha_gradient_logging)
# for i in range(10):
#     gumbel = functional.gumbel_softmax(logits = a, hard=True, tau=10)
#     for p in [gumbel, a]:
#         if p.grad is not None:
#             p.grad.detach_()
#             p.grad.zero_()
#     gumbel.max().backward()
# print("done")
# g = graph.GraphGenerator(3)
# graph = g.graphs[3]
# graphNN = diffNN.GraphNN(graph, 10)
# i = torch.ones(10)
# print(graphNN(i))