import sys
import argparse

from tqdm import tqdm
import yaml

sys.path.append("src")

from inference.inference_utils import prepare_translator
from data.data_utils import (
    read_text_sentences,
    get_metadata_languages_indexes,
    write_translation_output,
)
from data.tokenizer import MTTokenizer
from models.transformer import Transformer
from decoding.translator import Translator
from data.scoring import calculate_bleu_score


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
    translator = prepare_translator(model_parameters, params)
    source_sentences, target_sentences = _prepare_evaluation_data(model_parameters)
    translated_sentences = _translate_sentences(translator, source_sentences)
    _evaluate_bleu_score(translated_sentences, target_sentences)
    if params.output_path:
        write_translation_output(
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
