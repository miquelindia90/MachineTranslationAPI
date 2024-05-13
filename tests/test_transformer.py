import sys

import torch

sys.path.append("./src")

from models.encoder import Encoder
from models.decoder import Decoder
from models.transformer import Transformer


def test_encoder_parameter_count():

    encoder = Encoder(
        encoder_vocabulary_size=200,
        max_length=100,
        model_dimension=256,
        hidden_dimension=512,
        number_of_heads=8,
        number_of_layers=2,
    )

    total_params = sum(p.numel() for p in encoder.parameters())
    assert total_params == 1105408


def test_encoder_forward_pass():

    encoder = Encoder(
        encoder_vocabulary_size=500,
        max_length=100,
        model_dimension=256,
        hidden_dimension=512,
        number_of_heads=8,
        number_of_layers=2,
    )

    input_tensor = torch.zeros(10, 30).long()
    output = encoder(input_tensor, None)
    assert output.shape == (10, 30, 256)


def test_decoder_parameter_count():

    decoder = Decoder(
        decoder_vocabulary_size=200,
        max_length=100,
        model_dimension=256,
        hidden_dimension=512,
        number_of_heads=8,
        number_of_layers=2,
    )

    total_params = sum(p.numel() for p in decoder.parameters())
    assert total_params == 1684168


def test_decoder_forward_pass():

    decoder = Decoder(
        decoder_vocabulary_size=200,
        max_length=100,
        model_dimension=256,
        hidden_dimension=512,
        number_of_heads=8,
        number_of_layers=2,
    )

    target_tensor = torch.zeros(10, 30).long()
    encoder_source_tensor = torch.zeros(10, 30, 256)

    output = decoder(target_tensor, encoder_source_tensor, None, None)
    assert output.shape == (10, 30, 200)

def test_transformer_parameter_count():

    transformer = Transformer(
        source_padding_index=499,
        target_padding_index=199,
        target_sos_index=0,
        encoder_vocabulary_size=500,
        decoder_vocabulary_size=200,
        max_length=100,
        model_dimension=256,
        hidden_dimension=512,
        number_of_heads=8,
        number_of_layers=2,
    )

    total_params = sum(p.numel() for p in transformer.parameters())
    assert total_params == 2866376

def test_transformer_forward_pass():

    transformer = Transformer(
        source_padding_index=499,
        target_padding_index=199,
        target_sos_index=0,
        encoder_vocabulary_size=500,
        decoder_vocabulary_size=200,
        max_length=100,
        model_dimension=256,
        hidden_dimension=512,
        number_of_heads=8,
        number_of_layers=2,
    )

    source = torch.zeros(10, 30).long()
    target = torch.zeros(10, 30).long()

    output = transformer(source, target)
    assert output.shape == (10, 30, 200)