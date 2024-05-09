import sys
sys.path.append("./src")

from data_utils import DataIterator

def test_generator_initialization():
    data_iterator = DataIterator("examples/test.src", "examples/test.tgt", "examples/test.metadata", "en-sv")