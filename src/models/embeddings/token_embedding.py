from torch import nn


class TokenEmbedding(nn.Embedding):
    """
    TokenEmbedding class extends the nn.Embedding class and represents a token embedding layer.

    Args:
        vocablulary_size (int): The size of the vocabulary.
        model_dimension (int): The dimensionality of the token embeddings.
    """

    def __init__(self, vocablulary_size: int, model_dimension: int):
        super(TokenEmbedding, self).__init__(vocablulary_size, model_dimension, padding_idx=1)