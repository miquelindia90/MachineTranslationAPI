import math

import torch
from torch import nn


class ScaleDotProductAttention(nn.Module):
    """
    ScaleDotProductAttention module performs scaled dot-product attention.
    It takes query, key, and value tensors as input and returns the attention output and attention scores.
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        query: torch.tensor,
        key: torch.tensor,
        value: torch.tensor,
        mask: torch.tensor = None,
        epsilon: float = 1e-12,
    ) -> tuple:
        """
        Forward pass of the ScaleDotProductAttention module.

        Args:
            query (torch.Tensor): The query tensor.
            key (torch.Tensor): The key tensor.
            value (torch.Tensor): The value tensor.
            mask (torch.Tensor, optional): The mask tensor to apply on the attention scores. Defaults to None.
            epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-12.

        Returns:
            torch.Tensor: The attention output tensor.
            torch.Tensor: The attention scores tensor.
        """
        bach_size, number_of_heads, sequence_length, hidden_tensor = key.size()

        transposed_key = key.transpose(2, 3)
        score = (query @ transposed_key) / math.sqrt(hidden_tensor)

        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        score = self.softmax(score)

        value = score @ value

        return value, score
