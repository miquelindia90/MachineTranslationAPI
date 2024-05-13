import torch
from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """
    Decoder layer of a transformer model.

    Args:
        model_dimension (int): The dimension of the model.
        hidden_dimension (int): The dimension of the hidden layer in the position-wise feed forward network.
        number_of_heads (int): The number of attention heads.
        drop_probability (float): The dropout probability.

    Attributes:
        self_attention (MultiHeadAttention): The self-attention layer.
        norm1 (LayerNorm): The layer normalization after the self-attention layer.
        dropout1 (nn.Dropout): The dropout layer after the self-attention layer.
        encoder_decoder_attention (MultiHeadAttention): The encoder-decoder attention layer.
        norm2 (LayerNorm): The layer normalization after the encoder-decoder attention layer.
        dropout2 (nn.Dropout): The dropout layer after the encoder-decoder attention layer.
        hidden_layer (PositionwiseFeedForward): The position-wise feed forward network.
        norm3 (LayerNorm): The layer normalization after the position-wise feed forward network.
        dropout3 (nn.Dropout): The dropout layer after the position-wise feed forward network.

    Methods:
        forward(decoder_tensor, encoder_tensor, target_mask, source_mask): Performs the forward pass of the decoder layer.

    """

    def __init__(
        self,
        model_dimension: int,
        hidden_dimension: int,
        number_of_heads: int,
        drop_probability: float = 0.1,
    ):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(
            model_dimension=model_dimension, number_of_heads=number_of_heads
        )
        self.norm1 = LayerNorm(model_dimension=model_dimension)
        self.dropout1 = nn.Dropout(p=drop_probability)

        self.encoder_decoder_attention = MultiHeadAttention(
            model_dimension=model_dimension, number_of_heads=number_of_heads
        )
        self.norm2 = LayerNorm(model_dimension=model_dimension)
        self.dropout2 = nn.Dropout(p=drop_probability)

        self.hidden_layer = PositionwiseFeedForward(
            model_dimension=model_dimension,
            hidden_dimension=hidden_dimension,
            drop_probability=drop_probability,
        )
        self.norm3 = LayerNorm(model_dimension=model_dimension)
        self.dropout3 = nn.Dropout(p=drop_probability)

    def forward(
        self,
        decoder_tensor: torch.Tensor,
        encoder_tensor: torch.Tensor,
        target_mask: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs the forward pass of the decoder layer.

        Args:
            decoder_tensor (torch.Tensor): The input tensor to the decoder layer.
            encoder_tensor (torch.Tensor): The input tensor from the encoder layer (optional).
            target_mask (torch.Tensor): The mask for the decoder self-attention.
            source_mask (torch.Tensor): The mask for the encoder-decoder attention.

        Returns:
            torch.Tensor: The output tensor from the decoder layer.

        """
        # 1. compute self attention
        _x = decoder_tensor
        x = self.self_attention(
            query=decoder_tensor,
            key=decoder_tensor,
            value=decoder_tensor,
            mask=target_mask,
        )

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if encoder_tensor is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x = self.encoder_decoder_attention(
                query=x, key=encoder_tensor, value=encoder_tensor, mask=source_mask
            )

            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.hidden_layer(x)

        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x
