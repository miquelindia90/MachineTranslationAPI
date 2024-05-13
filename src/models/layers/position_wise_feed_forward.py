import torch
from torch import nn


class PositionwiseFeedForward(nn.Module):
    """
    Positionwise Feed Forward module in a Transformer model.

    Args:
        model_dimension (int): The input dimension of the module.
        hidden_dimension (int): The hidden dimension of the module.
        drop_probability (float, optional): The dropout probability. Default is 0.1.

    Attributes:
        linear1 (nn.Linear): The first linear layer.
        linear2 (nn.Linear): The second linear layer.
        relu (nn.ReLU): The ReLU activation function.
        dropout (nn.Dropout): The dropout layer.

    """

    def __init__(
        self, model_dimension: int, hidden_dimension: int, drop_probability: float = 0.1
    ):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(model_dimension, hidden_dimension)
        self.linear2 = nn.Linear(hidden_dimension, model_dimension)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PositionwiseFeedForward module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
