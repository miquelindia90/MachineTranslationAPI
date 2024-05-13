import torch
from torch import nn

from models.layers.scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention module that performs multi-head attention mechanism.

    Args:
        model_dimension (int): The dimension of the input and output tensors.
        number_of_heads (int): The number of attention heads.

    Attributes:
        number_of_heads (int): The number of attention heads.
        attention (ScaleDotProductAttention): The attention mechanism.
        weight_query (nn.Linear): Linear layer for query transformation.
        weight_key (nn.Linear): Linear layer for key transformation.
        weight_value (nn.Linear): Linear layer for value transformation.
        weight_concat (nn.Linear): Linear layer for concatenation.

    """

    def __init__(self, model_dimension: int, number_of_heads: int):
        super(MultiHeadAttention, self).__init__()
        self.number_of_heads = number_of_heads
        self.attention = ScaleDotProductAttention()
        self.weight_query = nn.Linear(model_dimension, model_dimension)
        self.weight_key = nn.Linear(model_dimension, model_dimension)
        self.weight_value = nn.Linear(model_dimension, model_dimension)
        self.weight_concat = nn.Linear(model_dimension, model_dimension)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of the MultiHeadAttention module.

        Args:
            query (torch.Tensor): The query tensor.
            key (torch.Tensor): The key tensor.
            value (torch.Tensor): The value tensor.
            maskey (torch.Tensor, optional): The mask tensor for masking out certain positions.

        Returns:
            torch.Tensor: The output tensor after multi-head attention mechanism.

        """

        query, key, value = (
            self.weight_query(query),
            self.weight_key(key),
            self.weight_value(value),
        )

        query, key, value = self.split(query), self.split(key), self.split(value)

        out, attention = self.attention(query, key, value, mask=mask)

        out = self.concat(out)
        out = self.weight_concat(out)

        return out

    def split(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Split the tensor into multiple heads.

        Args:
            tensor (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The tensor after splitting.

        """
        batch_size, length, model_dimension = tensor.size()

        tensor_dimension = model_dimension // self.number_of_heads
        tensor = tensor.view(
            batch_size, length, self.number_of_heads, tensor_dimension
        ).transpose(1, 2)

        return tensor

    def concat(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Concatenate the tensor from multiple heads.

        Args:
            tensor (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The tensor after concatenation.

        """
        batch_size, head, length, tensor_dimension = tensor.size()
        model_dimension = head * tensor_dimension

        tensor = (
            tensor.transpose(1, 2)
            .contiguous()
            .view(batch_size, length, model_dimension)
        )
        return tensor
