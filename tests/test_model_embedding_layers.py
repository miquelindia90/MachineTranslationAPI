import sys
import pytest

import torch
import matplotlib.pyplot as plt

sys.path.append("./src")

from models.embeddings.positional_encoding import PositionalEncoding

def test_positional_encoding_shape():
    positional_encoding = PositionalEncoding(model_dimension=512, max_length=100)
    input_tensor = torch.zeros((64, 50))
    output = positional_encoding(input_tensor)
    assert output.shape == (50, 512)
    with pytest.raises(ValueError) as exc_info:
        input_tensor = torch.zeros((64, 150))
        output = positional_encoding(input_tensor)
    assert str(exc_info.value) == "Input sequence length 150 is greater than maximum length 100."    


def test_positional_encoding_values():
    positional_encoding = PositionalEncoding(model_dimension=512, max_length=100)
    input_tensor = torch.zeros((64, 100))
    output = positional_encoding(input_tensor)
    assert output[0, 0] == 0.0
    assert output[0, 1] == 1.0
    assert round(output[25, 3].item(),4) == 0.5266
    assert round(output[70, 2].item(),4) == -0.9998
    plt.imshow(output, cmap='hot', interpolation='nearest')
    plt.savefig("tests/examples/positional_encoding.png")
    plt.close()

def test_positional_encoding_non_trainable():
    positional_encoding = PositionalEncoding(model_dimension=512, max_length=100)
    input_tensor = torch.zeros((64, 100))
    output = positional_encoding(input_tensor)
    assert not output.requires_grad

