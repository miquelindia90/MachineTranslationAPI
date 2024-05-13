import torch
from torch import nn


class LayerNorm(nn.Module):
    """
    Layer normalization module.

    Args:
        model_dimension (int): The input dimension of the layer.
        epsilon (float, optional): A small value added to the denominator for numerical stability. Default is 1e-12.

    Attributes:
        gamma (torch.nn.Parameter): The learnable scale parameter.
        beta (torch.nn.Parameter): The learnable shift parameter.
        epsilon (float): The small value added to the denominator for numerical stability.

    """

    def __init__(self, model_dimension, epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(model_dimension))
        self.beta = nn.Parameter(torch.zeros(model_dimension))
        self.epsilon = epsilon

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass of the LayerNorm module.

        Args:
            x (torch.tensor): Input tensor.

        Returns:
            torch.tensor: Output tensor after applying LayerNorm.

        """
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)

        out = (x - mean) / torch.sqrt(var + self.epsilon)
        out = self.gamma * out + self.beta
        return out