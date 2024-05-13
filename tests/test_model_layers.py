import sys

import torch

sys.path.append("./src")

from models.layers.layer_norm import LayerNorm
from models.layers.scale_dot_product_attention import ScaleDotProductAttention

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

def test_scale_dot_product_attention_forward():
    
    batch_size = 5
    number_of_heads = 8
    sequence_length = 30
    hidden_tensor_dimension = 10

    query = torch.randn(batch_size, number_of_heads, sequence_length, hidden_tensor_dimension)
    key = torch.randn(batch_size, number_of_heads, sequence_length, hidden_tensor_dimension)
    value = torch.randn(batch_size, number_of_heads, sequence_length,  hidden_tensor_dimension)

    scale_dot_product_attention = ScaleDotProductAttention()
    output_tensor, attention_scores = scale_dot_product_attention(query, key, value)
    assert output_tensor.shape == (batch_size, number_of_heads, sequence_length, hidden_tensor_dimension)
    assert attention_scores.shape == (batch_size, number_of_heads, sequence_length, sequence_length)
    # TO DO: Add more tests to check the correctness of the ScaleDotProductAttention module. Add tests to check the attention scores and the output tensor.