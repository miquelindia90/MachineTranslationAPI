import sys

import torch
from torch import nn

sys.path.append("./src")

from models.layers.decoder_layer import DecoderLayer
from models.embeddings.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    """
    The Decoder module of the Transformer model for machine translation.

    Args:
        decoder_vocabulary_size (int): The size of the decoder vocabulary.
        max_length (int): The maximum length of the input sequence.
        model_dimension (int): The dimension of the model.
        hidden_dimension (int): The dimension of the hidden layer.
        number_of_heads (int): The number of attention heads.
        number_of_layers (int): The number of decoder layers.
        drop_probability (float, optional): The probability of dropout. Defaults to 0.1.
        device (torch.device, optional): The device to run the model on. Defaults to torch.device("cpu").

    Attributes:
        emb (TransformerEmbedding): The embedding layer.
        layers (nn.ModuleList): The list of decoder layers.
        linear (nn.Linear): The linear layer for the language model head.

    """

    def __init__(
        self,
        decoder_vocabulary_size: int,
        max_length: int,
        model_dimension: int,
        hidden_dimension: int,
        number_of_heads: int,
        number_of_layers: int,
        drop_probability: float = 0.1,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.emb = TransformerEmbedding(
            model_dimension=model_dimension,
            drop_probability=drop_probability,
            max_length=max_length,
            vocabulary_size=decoder_vocabulary_size,
            device=device,
        )

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    model_dimension=model_dimension,
                    hidden_dimension=hidden_dimension,
                    number_of_heads=number_of_heads,
                    drop_probability=drop_probability,
                )
                for _ in range(number_of_layers)
            ]
        )

        self.linear = nn.Linear(model_dimension, decoder_vocabulary_size)

    def forward(
        self,
        target_tensor: torch.Tensor,
        encoder_source_tensor: torch.Tensor,
        target_mask: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the Decoder module.

        Args:
            target_tensor (torch.Tensor): The input tensor for the target sequence.
            encoder_source_tensor (torch.Tensor): The input tensor for the encoder source sequence.
            target_mask (torch.Tensor): The mask for the target sequence.
            source_mask (torch.Tensor): The mask for the encoder source sequence.

        Returns:
            torch.Tensor: The output tensor of the Decoder module.

        """
        target_tensor = self.emb(target_tensor)

        for layer in self.layers:
            target_tensor = layer(
                target_tensor, encoder_source_tensor, target_mask, source_mask
            )

        # pass to LM head
        output = self.linear(target_tensor)
        return output
