import sys
import torch
from torch import nn

sys.path.append("./src")
from models.embeddings.positional_encoding import PositionalEncoding
from models.embeddings.token_embedding import TokenEmbedding


class TransformerEmbedding(nn.Module):
    """
    TransformerEmbedding class represents the embedding layer in a Transformer model.

    Args:
        vocablulary_size (int): The size of the vocabulary.
        model_dimension (int): The dimension of the model.
        max_length (int): The maximum length of the input sequence.
        drop_probability (float): The probability of an element to be zeroed in the dropout layer.
        device (torch.device): The device on which the model is being trained.

    Attributes:
        token_embedding (TokenEmbedding): The token embedding layer.
        position_embedding (PositionalEncoding): The positional encoding layer.
        drop_out (nn.Dropout): The dropout layer.
        drop_probability (float): The probability of an element to be zeroed in the dropout layer.
    """

    def __init__(
        self,
        vocablulary_size: int,
        model_dimension: int,
        max_length: int,
        drop_probability: float,
        device: torch.device,
    ):
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(vocablulary_size, model_dimension)
        self.position_embedding = PositionalEncoding(
            model_dimension, max_length, device
        )
        self.drop_out = nn.Dropout(p=drop_probability)
        self.drop_probability = drop_probability

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass of the TransformerEmbedding.

        Args:
            x (torch.tensor): The input tensor.

        Returns:
            torch.tensor: The output tensor after applying token and positional embeddings.
        """
        token_embedding = self.token_embedding(x)
        position_embedding = self.position_embedding(x)
        if self.drop_probability > 0.0:
            return self.drop_out(token_embedding + position_embedding)
        else:
            return token_embedding + position_embedding
