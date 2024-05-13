import sys

import torch

sys.path.append("./src")

from models.layers.layer_norm import LayerNorm
from models.layers.scale_dot_product_attention import ScaleDotProductAttention
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward
from models.layers.encoder_layer import EncoderLayer
from models.layers.decoder_layer import DecoderLayer


def test_layer_norm_forward():
    input_size = 10
    batch_size = 5
    epsilon = 1e-5

    input_tensor = torch.randn(batch_size, input_size)
    layer_norm = LayerNorm(input_size, epsilon=epsilon)
    output_tensor = layer_norm(input_tensor)
    assert output_tensor.shape == (batch_size, input_size)
    assert torch.allclose(
        output_tensor.mean(dim=1), torch.zeros(batch_size), atol=epsilon
    )
    # TO DO: Add more tests to check the correctness of the LayerNorm module. Add tests to check the variance of the output tensor.


def test_scale_dot_product_attention_forward():

    batch_size = 5
    number_of_heads = 8
    sequence_length = 30
    hidden_tensor_dimension = 10

    query = torch.randn(
        batch_size, number_of_heads, sequence_length, hidden_tensor_dimension
    )
    key = torch.randn(
        batch_size, number_of_heads, sequence_length, hidden_tensor_dimension
    )
    value = torch.randn(
        batch_size, number_of_heads, sequence_length, hidden_tensor_dimension
    )

    scale_dot_product_attention = ScaleDotProductAttention()
    output_tensor, attention_scores = scale_dot_product_attention(query, key, value)
    assert output_tensor.shape == (
        batch_size,
        number_of_heads,
        sequence_length,
        hidden_tensor_dimension,
    )
    assert attention_scores.shape == (
        batch_size,
        number_of_heads,
        sequence_length,
        sequence_length,
    )
    # TO DO: Add more tests to check the correctness of the ScaleDotProductAttention module. Add tests to check the attention scores and the output tensor.


def test_multi_head_attention_forward():
    batch_size = 5
    number_of_heads = 8
    sequence_length = 30
    hidden_tensor_dimension = 32

    query = torch.randn(batch_size, sequence_length, hidden_tensor_dimension)
    key = torch.randn(batch_size, sequence_length, hidden_tensor_dimension)
    value = torch.randn(batch_size, sequence_length, hidden_tensor_dimension)

    multi_head_attention = MultiHeadAttention(hidden_tensor_dimension, number_of_heads)
    output_tensor = multi_head_attention(query, key, value)
    assert output_tensor.shape == (batch_size, sequence_length, hidden_tensor_dimension)


def test_position_wise_feed_forward_forward():
    batch_size = 5
    sequence_length = 30
    model_dimension = 32
    hidden_dimension = 64

    input_tensor = torch.randn(batch_size, sequence_length, model_dimension)
    positionwise_feed_forward = PositionwiseFeedForward(
        model_dimension, hidden_dimension
    )
    output_tensor = positionwise_feed_forward(input_tensor)
    assert output_tensor.shape == (batch_size, sequence_length, model_dimension)


def test_encoder_layer_forward():
    batch_size = 5
    sequence_length = 30
    model_dimension = 32
    hidden_dimension = 64
    number_of_heads = 8

    x = torch.randn(batch_size, sequence_length, model_dimension)

    encoder_layer = EncoderLayer(model_dimension, hidden_dimension, number_of_heads)
    output_tensor = encoder_layer(x, None)

    assert output_tensor.shape == (batch_size, sequence_length, model_dimension)
    # TO DO: Add more tests to check the correctness of the EncoderLayer module. Add tests to chech the src_mask argument.


def test_decoder_layer_forward():
    batch_size = 5
    sequence_length = 30
    model_dimension = 32
    hidden_dimension = 64
    number_of_heads = 8

    x = torch.randn(batch_size, sequence_length, model_dimension)
    encoder_output = torch.randn(batch_size, sequence_length, model_dimension)

    decoder_layer = DecoderLayer(model_dimension, hidden_dimension, number_of_heads)
    output_tensor = decoder_layer(x, encoder_output, None, None)

    assert output_tensor.shape == (batch_size, sequence_length, model_dimension)
