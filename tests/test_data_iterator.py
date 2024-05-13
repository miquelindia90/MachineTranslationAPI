import sys

sys.path.append("./src")

from data_iterator import DataIterator
from tokenizer import MTTokenizer

def test_generator_initialization():
    test_source_path = "tests/examples/test.src"
    test_target_path = "tests/examples/test.tgt"
    test_metadata_path = "tests/examples/test.metadata"
    language_pairs = ["None", '"src_lang": "en", "tgt_lang": "sv"', '"src_lang": "nb", "tgt_lang": "da"', '"src_lang": "sv", "tgt_lang": "nb"']
    data_iterator_lengths = [1000, 83, 108, 100]

    for language_pair, data_iterator_length in zip(language_pairs, data_iterator_lengths):
        tokenizer = MTTokenizer()    
        data_iterator = DataIterator(
        test_source_path,
        test_target_path,
        test_metadata_path,
        language_pair
        )
        assert len(data_iterator) == data_iterator_length

    
