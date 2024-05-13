import sys

import torch
from torch import nn

sys.path.append("./source")

from models.decoder import Decoder
from models.encoder import Encoder


class Transformer(nn.Module):
    """
    Transformer model for machine translation.

    Args:
        source_padding_index (int): Index of the padding token in the source vocabulary.
        target_padding_index (int): Index of the padding token in the target vocabulary.
        target_sos_index (int): Index of the start-of-sequence token in the target vocabulary.
        encoder_vocabulary_size (int): Size of the source vocabulary.
        decoder_vocabulary_size (int): Size of the target vocabulary.
        model_dimension (int): Dimensionality of the model.
        number_of_heads (int): Number of attention heads.
        max_length (int): Maximum sequence length.
        hidden_dimension (int): Dimensionality of the hidden layers.
        number_of_layers (int): Number of layers in the encoder and decoder.
        drop_probability (float, optional): Dropout probability. Defaults to 0.1.
        device (torch.device, optional): Device to run the model on. Defaults to torch.device("cpu").

    Attributes:
        source_padding_index (int): Index of the padding token in the source vocabulary.
        target_padding_index (int): Index of the padding token in the target vocabulary.
        target_sos_index (int): Index of the start-of-sequence token in the target vocabulary.
        device (torch.device): Device to run the model on.
        encoder (Encoder): Encoder module of the transformer.
        decoder (Decoder): Decoder module of the transformer.

    """

    def __init__(
        self,
        source_padding_index: int,
        target_padding_index: int,
        target_sos_index: int,
        encoder_vocabulary_size: int,
        decoder_vocabulary_size: int,
        model_dimension: int,
        number_of_heads: int,
        max_length: int,
        hidden_dimension: int,
        number_of_layers: int,
        drop_probability: float = 0.1,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.source_padding_index = source_padding_index
        self.target_padding_index = target_padding_index
        self.target_sos_index = target_sos_index
        self.device = device
        self.encoder = Encoder(
            model_dimension=model_dimension,
            number_of_heads=number_of_heads,
            max_length=max_length,
            hidden_dimension=hidden_dimension,
            encoder_vocabulary_size=encoder_vocabulary_size,
            drop_probability=drop_probability,
            number_of_layers=number_of_layers,
            device=device,
        )

        self.decoder = Decoder(
            model_dimension=model_dimension,
            number_of_heads=number_of_heads,
            max_length=max_length,
            hidden_dimension=hidden_dimension,
            decoder_vocabulary_size=decoder_vocabulary_size,
            drop_probability=drop_probability,
            number_of_layers=number_of_layers,
            device=device,
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer model.

        Args:
            source (torch.Tensor): Input source sequence tensor of shape (batch_size, source_length).
            target (torch.Tensor): Input target sequence tensor of shape (batch_size, target_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, target_length, decoder_vocabulary_size).

        """
        source_mask = self.make_source_mask(source)
        target_mask = self.make_target_mask(target)
        encoder_source = self.encoder(source, source_mask)
        output = self.decoder(target, encoder_source, target_mask, source_mask)
        return output

    def make_source_mask(self, source: torch.Tensor) -> torch.Tensor:
        """
        Create a mask for the source sequence.

        Args:
            source (torch.Tensor): Input source sequence tensor of shape (batch_size, source_length).

        Returns:
            torch.Tensor: Source mask tensor of shape (batch_size, 1, 1, source_length).

        """
        source_mask = (source != self.source_padding_index).unsqueeze(1).unsqueeze(2)
        return source_mask

    def make_target_mask(self, target: torch.Tensor) -> torch.Tensor:
        """
        Create a mask for the target sequence.

        Args:
            target (torch.Tensor): Input target sequence tensor of shape (batch_size, target_length).

        Returns:
            torch.Tensor: Target mask tensor of shape (batch_size, 1, target_length, target_length).

        """
        target_padding_mask = (
            (target != self.target_padding_index).unsqueeze(1).unsqueeze(3)
        )
        target_length = target.shape[1]
        target_sub_mask = (
            torch.tril(torch.ones(target_length, target_length))
            .type(torch.ByteTensor)
            .to(self.device)
        )
        target_mask = target_padding_mask & target_sub_mask
        return target_mask
