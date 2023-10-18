

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'


import torch
import torch.nn as nn
current = os.path.dirname(os.path.abspath(__file__))
os.chdir(current)

from models.n3net import N3Block
# x = N3Block(3,3) 

# x

# Test the N3Block
def test_n3block(device):
    # Create a random input tensor: [batch_size, channels, height, width]
    x = torch.randn(1, 3, 32, 32).to(device)
    
    # Create an N3Block instance
    n3block = N3Block(3,8).to(device)
    
    # Forward pass
    y = n3block(x)
    
    # Print output shape
    print("Output shape:", y.shape)

# Run the test
device = "cuda:0" if torch.cuda.is_available else "cpu"
test_n3block(device)
