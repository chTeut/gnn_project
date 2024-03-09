import torch 
import torch.nn as nn

m = nn.Linear(2, 3)
print(m.__doc__)
input = torch.randn(2, 2)
output = m(input)

print(output)

input = torch.randn(2, 2)
output = m(input)

print(output)