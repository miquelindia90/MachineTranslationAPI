import os
import json
import sys

sys.path.append("./src")

from data_utils import read_text_sentences, get_metadata_languages_indexes
from data_iterator import DataIterator
from tokenizer import SingleTokenizer, MTTokenizer


def test_read_text_sentences():
    
    file_path = "tests/examples/test.src"
    sentences = read_text_sentences(file_path)
    assert len(sentences) == 185407
    assert sentences[0] == "And it is also the case with the new national-conservative populism of today."
    assert sentences[1] == "Selv om du bruker en søkemotor, må du sørge for at den fører deg til det virkelige nettstedet."
    assert sentences[2] == "The links are maintained by their respective organisations, which are solely responsible for their content."

def test_SingleTokenizer_initialization():
    tokenizer = SingleTokenizer()
    assert tokenizer.get_tokens_dictionary() == dict()


def test_SingleTokenizer_training_model():
    tokenizer = SingleTokenizer()
    tokenizer.train(
        "tests/examples/test.src", "tests/examples/test.metadata", '"src_lang": "nb"'
    )
    tokenizer_dictionary = tokenizer.get_tokens_dictionary()
    assert tokenizer_dictionary["SOS"] == 0
    assert tokenizer_dictionary["UNK"] == 104452
    tokenizer.save_tokens_dictionary("tests/examples/output_tokens.json")
    with open("tests/examples/reference_tokens.json", "r") as target_file, open(
        "tests/examples/output_tokens.json", "r"
    ) as output_file:
        target_dict = json.load(target_file)
        output_dict = json.load(output_file)
        assert target_dict == output_dict
    os.remove("tests/examples/output_tokens.json")


def test_SingleTokenizer_inference_model():
    tokenizer = SingleTokenizer()
    tokenizer.load_tokens_dictionary("tests/examples/reference_tokens.json")
    assert tokenizer.word_to_id("UNK") == 104452
    assert tokenizer.word_to_id("SOS") == 0
    assert tokenizer.sentence_to_id_list("Restauranter Asiatisk i Minto Road") == [
        88,
        89,
        64,
        90,
        91,
    ]


def test_MTTokenizer_initialization():
    tokenizer = MTTokenizer()
    assert tokenizer.get_source_tokens_dictionary() == dict()
    assert tokenizer.get_target_tokens_dictionary() == dict()


# def test_MTTokenizer_training_model():
#     tokenizer = MTTokenizer()
#     tokenizer.train("tests/examples/test.src", "tests/examples/test.tgt")


# def test_MTTokenizer_inference_model():
#     tokenizer = MTTokenizer()


def test_generator_initialization():
    data_iterator = DataIterator(
        "tests/examples/test.src",
        "tests/examples/test.tgt",
        "tests/examples/test.metadata",
        "None",
    )
    assert len(data_iterator) == 185407
    data_iterator = DataIterator(
        "tests/examples/test.src",
        "tests/examples/test.tgt",
        "tests/examples/test.metadata",
        '"src_lang": "en", "tgt_lang": "sv"',
    )
    assert len(data_iterator) == 17198
    data_iterator = DataIterator(
        "tests/examples/test.src",
        "tests/examples/test.tgt",
        "tests/examples/test.metadata",
        '"src_lang": "nb", "tgt_lang": "da"',
    )
    assert len(data_iterator) == 19395
    data_iterator = DataIterator(
        "tests/examples/test.src",
        "tests/examples/test.tgt",
        "tests/examples/test.metadata",
        '"src_lang": "sv", "tgt_lang": "nb"',
    )
    assert len(data_iterator) == 19099
