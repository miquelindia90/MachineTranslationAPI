import torch
from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """
    This class represents an encoder layer in a transformer model.

    Args:
        model_dimension (int): The dimension of the input and output tensors.
        hidden_dimension (int): The dimension of the hidden layer in the positionwise feed forward network.
        number_of_heads (int): The number of attention heads in the multi-head attention mechanism.
        drop_probability (float): The probability of dropping elements during dropout.

    Attributes:
        attention (MultiHeadAttention): The multi-head attention module.
        norm1 (LayerNorm): The layer normalization module after the first attention layer.
        dropout1 (nn.Dropout): The dropout layer after the first attention layer.
        hidden_layer (PositionwiseFeedForward): The positionwise feed forward network.
        norm2 (LayerNorm): The layer normalization module after the hidden layer.
        dropout2 (nn.Dropout): The dropout layer after the hidden layer.
    """

    def __init__(
        self,
        model_dimension: int,
        hidden_dimension: int,
        number_of_heads: int,
        drop_probability: float = 0.1,
    ):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(
            model_dimension=model_dimension, number_of_heads=number_of_heads
        )
        self.norm1 = LayerNorm(model_dimension=model_dimension)
        self.dropout1 = nn.Dropout(p=drop_probability)

        self.hidden_layer = PositionwiseFeedForward(
            model_dimension=model_dimension,
            hidden_dimension=hidden_dimension,
            drop_probability=drop_probability,
        )
        self.norm2 = LayerNorm(model_dimension=model_dimension)
        self.dropout2 = nn.Dropout(p=drop_probability)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the encoder layer.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, model_dimension).
            src_mask (torch.Tensor): The source mask tensor of shape (batch_size, sequence_length, sequence_length).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, sequence_length, model_dimension).
        """
        # 1. compute self attention
        _x = x
        x = self.attention(query=x, key=x, value=x, mask=src_mask)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 3. positionwise feed forward network
        _x = x
        x = self.hidden_layer(x)

        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x
