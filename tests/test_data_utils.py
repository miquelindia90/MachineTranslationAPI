import sys

sys.path.append("./src")

from data_utils import DataIterator, SingleTokenizer, MTTokenizer


def test_SingleTokenizer_initialization():
    tokenizer = SingleTokenizer()
    assert tokenizer.get_tokens_dictionary() == dict()


def test_SingleTokenizer_training_model():
    tokenizer = SingleTokenizer()
    tokenizer.train("tests/examples/test.src")


def test_SingleTokenizer_inference_model():
    tokenizer = SingleTokenizer()


def test_MTTokenizer_initialization():
    tokenizer = MTTokenizer()
    assert tokenizer.get_source_tokens_dictionary() == dict()
    assert tokenizer.get_target_tokens_dictionary() == dict()


def test_MTTokenizer_training_model():
    tokenizer = MTTokenizer()
    tokenizer.train("tests/examples/test.src", "tests/examples/test.tgt")


def test_MTTokenizer_inference_model():
    tokenizer = MTTokenizer()


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
        "en-sv",
    )
    assert len(data_iterator) == 17198
    data_iterator = DataIterator(
        "tests/examples/test.src",
        "tests/examples/test.tgt",
        "tests/examples/test.metadata",
        "nb-da",
    )
    assert len(data_iterator) == 19395
    data_iterator = DataIterator(
        "tests/examples/test.src",
        "tests/examples/test.tgt",
        "tests/examples/test.metadata",
        "sv-nb",
    )
    assert len(data_iterator) == 19099
