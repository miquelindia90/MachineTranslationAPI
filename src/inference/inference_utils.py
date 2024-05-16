import sys

import torch

sys.path.append("./src")

from data.tokenizer import MTTokenizer
from models.transformer import Transformer
from decoding.translator import Translator


def load_tokenizer(model_path: str) -> MTTokenizer:
    """
    Load the tokenizer from the given model path.

    Args:
        model_path (str): The path to the model.

    Returns:
        MTTokenizer: The loaded tokenizer.
    """
    tokenizer = MTTokenizer()
    tokenizer.load_tokens_dictionary(
        model_path + "/source_tokens.json", model_path + "/target_tokens.json"
    )
    return tokenizer


def load_model(
    model_parameters: dict,
    tokenizer: MTTokenizer,
    model_path: str,
    device: torch.device = torch.device("cpu"),
) -> Transformer:
    """
    Load a pre-trained Transformer model from the given model path.

    Args:
        model_parameters (dict): A dictionary containing the model parameters.
        tokenizer (MTTokenizer): An instance of the MTTokenizer class.
        model_path (str): The path to the pre-trained model.
        device (torch.device, optional): The device to load the model on. Defaults to torch.device("cpu").

    Returns:
        Transformer: The loaded Transformer model.

    """
    model = Transformer(
        source_padding_index=tokenizer.source_lang_word_to_id("PAD"),
        target_padding_index=tokenizer.target_lang_word_to_id("PAD"),
        target_sos_index=tokenizer.target_lang_word_to_id("SOS"),
        encoder_vocabulary_size=len(tokenizer.get_source_tokens_dictionary()),
        decoder_vocabulary_size=len(tokenizer.get_target_tokens_dictionary()),
        model_dimension=model_parameters["model_dimension"],
        number_of_heads=model_parameters["number_of_heads"],
        max_length=model_parameters["max_length"],
        hidden_dimension=model_parameters["hidden_dimension"],
        number_of_layers=model_parameters["number_of_layers"],
        drop_probability=model_parameters["drop_probability"],
        device=device,
    )
    model.load_state_dict(
        torch.load(model_path, map_location=device)["model_state_dict"]
    )
    model.eval()
    return model


def prepare_translator(
    model_parameters: dict, model_path: str, device: str = "cpu"
) -> Translator:
    """
    Prepare the translator object for machine translation.

    Args:
        model_parameters (dict): A dictionary containing the model parameters.
        params (argparse.Namespace): An object containing the command line arguments.

    Returns:
        Translator: The prepared translator object.
    """
    tokenizer = load_tokenizer(model_path)
    model = load_model(
        model_parameters,
        tokenizer,
        model_path + "/model.pt",
        device=torch.device(device),
    )
    translator = Translator(model, tokenizer, device=torch.device(device))
    return translator


