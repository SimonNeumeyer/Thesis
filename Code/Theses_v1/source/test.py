import torch 
import torch.nn.functional as functional
from myLogging import alpha_gradient_logging

#a = torch.ones(5)
a = torch.tensor([-5,-1,2,3,4,5], dtype=torch.float32)
a.requires_grad = True
a.register_hook(alpha_gradient_logging)
for i in range(10):
    gumbel = functional.gumbel_softmax(logits = a, hard=True, tau=10)
    print(gumbel)
    for p in [gumbel, a]:
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()
    gumbel.max().backward()
print("done")
n = torch.multinomial(a, 1)
print(n)
n.backward()
y, i = torch.max(a, 0)
y.backward()
#torch.nn.functional.softmax(a).max().backward()
#a.max().backward()