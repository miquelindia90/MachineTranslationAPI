import sys

import torch
from torch import nn

sys.path.append("./src")

from models.layers.encoder_layer import EncoderLayer
from models.embeddings.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):
    """
    The Encoder class represents the encoder component of a Transformer model.

    Args:
        encoder_vocabulary_size (int): The size of the encoder vocabulary.
        max_length (int): The maximum length of the input sequence.
        model_dimension (int): The dimension of the model.
        hidden_dimension (int): The dimension of the hidden layer in the encoder.
        number_of_heads (int): The number of attention heads in the encoder.
        number_of_layers (int): The number of layers in the encoder.
        drop_probability (float, optional): The probability of dropout. Defaults to 0.1.
        device (torch.device, optional): The device to run the encoder on. Defaults to torch.device("cpu").

    Attributes:
        embedding_layer (TransformerEmbedding): The embedding layer of the encoder.
        layers (nn.ModuleList): The list of encoder layers.

    """

    def __init__(
        self,
        encoder_vocabulary_size: int,
        max_length: int,
        model_dimension: int,
        hidden_dimension: int,
        number_of_heads: int,
        number_of_layers: int,
        drop_probability: float = 0.1,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.embedding_layer = TransformerEmbedding(
            vocabulary_size=encoder_vocabulary_size,
            model_dimension=model_dimension,
            max_length=max_length,
            drop_probability=drop_probability,
            device=device,
        )

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    model_dimension=model_dimension,
                    hidden_dimension=hidden_dimension,
                    number_of_heads=number_of_heads,
                    drop_probability=drop_probability,
                )
                for _ in range(number_of_layers)
            ]
        )

    def forward(self, x: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length).
            source_mask (torch.Tensor): The source mask tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, sequence_length, model_dimension).

        """
        x = self.embedding_layer(x)

        for layer in self.layers:
            x = layer(x, source_mask)

        return x
