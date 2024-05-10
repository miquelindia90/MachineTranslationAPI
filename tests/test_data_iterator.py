import sys

sys.path.append("./src")

from data_iterator import DataIterator


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
