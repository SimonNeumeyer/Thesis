import diffNN
import graph
import torch
import torch.nn as nn
 
        
if __name__ == "__main__":
    module = nn.Identity()
    module2 = nn.Linear(10, 10)
    diffNN = diffNN.DiffNN([module, module2])
    x = torch.ones(10)
    print(diffNN(x))