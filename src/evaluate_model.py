import sys
import argparse

import torch
from tqdm import tqdm
import yaml

sys.path.append("src")

from data.data_utils import read_text_sentences, get_metadata_languages_indexes
from data.tokenizer import MTTokenizer
from models.transformer import Transformer
from decoding.translator import Translator
from data.scoring import calculate_bleu_score


def _load_tokenizer(model_path: str) -> MTTokenizer:
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


def _load_model(
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


def _prepare_translator(
    model_parameters: dict, params: argparse.Namespace
) -> Translator:
    """
    Prepare the translator object for machine translation.

    Args:
        model_parameters (dict): A dictionary containing the model parameters.
        params (argparse.Namespace): An object containing the command line arguments.

    Returns:
        Translator: The prepared translator object.
    """
    tokenizer = _load_tokenizer(params.model)
    model = _load_model(
        model_parameters,
        tokenizer,
        params.model + "/model.pt",
        device=torch.device(params.device),
    )
    translator = Translator(model, tokenizer, device=torch.device(params.device))
    return translator


def _prepare_evaluation_data(parameters: dict) -> tuple:
    """
    Prepare the evaluation data for the model.

    Args:
        parameters (dict): A dictionary containing the necessary parameters for evaluation.

    Returns:
        tuple: A tuple containing the indexed source sentences and indexed target sentences.
    """
    source_sentences = read_text_sentences(parameters["test_src_path"])
    target_sentences = read_text_sentences(parameters["test_tgt_path"])
    language_indexes = get_metadata_languages_indexes(
        parameters["test_metadata_path"], parameters["language_filter_str"]
    )
    indexed_source_sentences = [source_sentences[index] for index in language_indexes]
    indexed_target_sentences = [target_sentences[index] for index in language_indexes]
    return indexed_source_sentences, indexed_target_sentences


def _translate_sentences(translator: Translator, source_sentences: list) -> list:
    """
    Translates a list of source sentences using the given translator.

    Args:
        translator (Translator): The translator object used for translation.
        source_sentences (list): A list of source sentences to be translated.

    Returns:
        list: A list of translated sentences.

    """
    translated_senteces = list()
    for sentence in tqdm(source_sentences, desc="Translating Sentences"):
        translated_senteces.append(translator.translate(sentence))
    return translated_senteces


def _evaluate_bleu_score(translated_sentences: list, target_sentences: list) -> None:
    """
    Calculate and print the BLEU score for the translated sentences.

    Args:
        translated_sentences (list): A list of translated sentences.
        target_sentences (list): A list of target sentences.

    Returns:
        None
    """
    bleu = 0.0
    for translated_sentence, target_sentence in zip(
        translated_sentences, target_sentences
    ):
        bleu += calculate_bleu_score(translated_sentence, [target_sentence])
    print(f"BLEU score: {bleu/len(translated_sentences)}")


def _write_translation_output(
    source_sentences: list,
    translated_sentences: list,
    target_sentences: list,
    output_path: str,
) -> None:
    """
    Write the translation output to a file.

    Args:
        source_sentences (list): List of source sentences.
        translated_sentences (list): List of translated sentences.
        target_sentences (list): List of target sentences.
        output_path (str): Path to the output file.

    Returns:
        None
    """
    with open(output_path, "w") as output_file:
        for source_sentence, translated_sentence, target_sentence in zip(
            source_sentences, translated_sentences, target_sentences
        ):
            output_file.write(f"Src: {source_sentence}\n")
            output_file.write(f"Tra: {translated_sentence}\n")
            output_file.write(f"Tgt: {target_sentence}\n\n")


def main(params: argparse.Namespace) -> None:
    """
    Main function for evaluating a machine translation model.

    Args:
        params (argparse.Namespace): Command-line arguments and parameters.

    Returns:
        None
    """
    print(f"Evaluating model at {params.model}")
    model_parameters = yaml.safe_load(open(params.model + "/config.yaml", "r"))
    translator = _prepare_translator(model_parameters, params)
    source_sentences, target_sentences = _prepare_evaluation_data(model_parameters)
    translated_sentences = _translate_sentences(translator, source_sentences)
    _evaluate_bleu_score(translated_sentences, target_sentences)
    if params.output_path:
        _write_translation_output(
            source_sentences, translated_sentences, target_sentences, params.output_path
        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Evaluate a model")
    argparser.add_argument(
        "--model", "-m", type=str, required=True, help="Path to the model"
    )
    argparser.add_argument(
        "--output_path", "-o", type=str, default="", help="Path to the output file"
    )
    argparser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cpu",
        choices=["cpu", "cuda:0"],
        help="Device to run the model on",
    )
    params = argparser.parse_args()
    main(params)
