import sys
sys.path.append("./src")

from data_utils import DataIterator

def test_generator_initialization():
    data_iterator = DataIterator("tests/examples/test.src", "tests/examples/test.tgt", "tests/examples/test.metadata", "None")
    assert len(data_iterator) == 185407
    data_iterator = DataIterator("tests/examples/test.src", "tests/examples/test.tgt", "tests/examples/test.metadata", "en-sv")
    assert len(data_iterator) == 17198
    data_iterator = DataIterator("tests/examples/test.src", "tests/examples/test.tgt", "tests/examples/test.metadata", "nb-da")
    assert len(data_iterator) == 19395
    data_iterator = DataIterator("tests/examples/test.src", "tests/examples/test.tgt", "tests/examples/test.metadata", "sv-nb")
    assert len(data_iterator) == 19099