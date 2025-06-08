import torch
import torch.nn.init as init
import math

def uniform_(tensor, scale=0.05):
    """Uniform init."""
    init.uniform_(tensor, a=-scale, b=scale)

def glorot_(tensor):
    """Glorot & Bengio (AISTATS 2010) init (Xavier uniform)."""
    init.xavier_uniform_(tensor)

def glorot_normal_(tensor):
    """Glorot & Bengio (AISTATS 2010) init (Xavier normal)."""
    init.xavier_normal_(tensor)

def zeros_(tensor):
    """All zeros."""
    init.zeros_(tensor)

def ones_(tensor):
    """All ones."""
    init.ones_(tensor)

# Example of how these might be used if a layer didn't initialize its own parameters:
# class MyLayer(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(MyLayer, self).__init__()
#         self.weight = torch.nn.Parameter(torch.Tensor(input_dim, output_dim))
#         self.bias = torch.nn.Parameter(torch.Tensor(output_dim))
#         self.reset_parameters()

#     def reset_parameters(self):
#         glorot_(self.weight)
#         zeros_(self.bias)

#     def forward(self, x):
#         # ...
#         return x
