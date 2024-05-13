import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    Positional Encoding module for Transformer models.

    Args:
        model_dimension (int): The dimension of the input embeddings.
        max_length (int): The maximum length of the input sequence.
        device (torch.device): The device on which the tensors will be allocated.

    Attributes:
        encoding (torch.Tensor): The positional encoding tensor of shape (max_length, model_dimension).
    """

    def __init__(self, model_dimension: int, max_length: int, device: torch.device=torch.device("cpu")):
        super(PositionalEncoding, self).__init__()
        
        self.max_length = max_length
        self.encoding = torch.zeros(max_length, model_dimension, device=device)
        self.encoding.requires_grad = False
        pos = torch.arange(0, max_length, device=device)
        pos = pos.float().unsqueeze(dim=1)

        even_indices = torch.arange(0, model_dimension, step=2, device=device).float()
        self.encoding[:, 0::2] = torch.sin(
            pos / (10000 ** (even_indices / model_dimension))
        )
        self.encoding[:, 1::2] = torch.cos(
            pos / (10000 ** (even_indices / model_dimension))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: The encoded tensor of shape (seq_len, model_dimension).

        Raises:
            ValueError: If the input sequence length is greater than the maximum length.
        """
        batch_size, seq_len = x.size()
        if seq_len > self.max_length:
            raise ValueError(
                f"Input sequence length {seq_len} is greater than maximum length {self.max_length}."
            )
        return self.encoding[:seq_len, :]
