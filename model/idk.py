import torch.nn as nn
import torch

loss = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
#L1 loss function parameters explanation applies here.

input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)
print(input.size(), target.size())
output = loss(input, target)
print(output) #tensor(0.9823, grad_fn=<MseLossBackward>)