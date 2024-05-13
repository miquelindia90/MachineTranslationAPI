import sys

import torch

sys.path.append("./src")

from models.layers.layer_norm import LayerNorm

def test_layer_norm_forward():
    input_size = 10
    batch_size = 5
    epsilon = 1e-5

    input_tensor = torch.randn(batch_size, input_size)
    layer_norm = LayerNorm(input_size, epsilon=epsilon)
    output_tensor = layer_norm(input_tensor)
    assert output_tensor.shape == (batch_size, input_size)
    assert torch.allclose(output_tensor.mean(dim=1), torch.zeros(batch_size), atol=epsilon)
    # TO DO: Add more tests to check the correctness of the LayerNorm module. Add tests to check the variance of the output tensor.