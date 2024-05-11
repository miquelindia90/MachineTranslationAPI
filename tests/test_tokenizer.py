import sys
import os
import json

sys.path.append("./src")

from tokenizer import SingleTokenizer, MTTokenizer

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
    assert tokenizer_dictionary["UNK"] == 1877
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
    assert tokenizer.word_to_id("UNK") == 1877
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


def test_MTTokenizer_training_model():
    tokenizer = MTTokenizer()
    tokenizer.train("tests/examples/test.src", "tests/examples/test.tgt", "tests/examples/test.metadata", '"src_lang": "en", "tgt_lang": "sv"')
    source_tokens_dictionary = tokenizer.get_source_tokens_dictionary()
    target_tokens_dictionary = tokenizer.get_target_tokens_dictionary()
    assert source_tokens_dictionary["SOS"] == 0
    assert source_tokens_dictionary["UNK"] == 668
    assert target_tokens_dictionary["SOS"] == 0
    assert target_tokens_dictionary["UNK"] == 675
    tokenizer.save_tokens_dictionary("tests/examples/output_source_tokens.json", "tests/examples/output_target_tokens.json")
    with open("tests/examples/reference_source_tokens.json", "r") as target_file, open("tests/examples/output_source_tokens.json", "r") as output_file:
        target_dict = json.load(target_file)
        output_dict = json.load(output_file)
        assert target_dict == output_dict
    os.remove("tests/examples/output_source_tokens.json")
    with open("tests/examples/reference_target_tokens.json", "r") as target_file, open("tests/examples/output_target_tokens.json", "r") as output_file:
        target_dict = json.load(target_file)
        output_dict = json.load(output_file)
        assert target_dict == output_dict
    os.remove("tests/examples/output_target_tokens.json")

def test_MTTokenizer_inference_model():
    tokenizer = MTTokenizer()
    tokenizer.load_tokens_dictionary("tests/examples/reference_source_tokens.json", "tests/examples/reference_target_tokens.json")
    assert tokenizer.source_lang_word_to_id("UNK") == 668
    assert tokenizer.source_lang_word_to_id("SOS") == 0
    assert tokenizer.target_lang_word_to_id("UNK") == 675
    assert tokenizer.target_lang_word_to_id("SOS") == 0
    assert tokenizer.source_lang_sentence_to_id_list("And it is also the case") == [1, 2, 3, 4, 5, 6]
    assert tokenizer.target_lang_sentence_to_id_list("Så är det även i dag med") == [1, 2, 3, 4, 5, 6, 7]