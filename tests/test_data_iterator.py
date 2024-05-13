import sys
import pytest

sys.path.append("./src")

from data_iterator import DataIterator
from tokenizer import MTTokenizer

def test_generator_initialization():
    test_source_path = "tests/examples/test.src"
    test_target_path = "tests/examples/test.tgt"
    test_metadata_path = "tests/examples/test.metadata"
    language_pair = '"src_lang": "en", "tgt_lang": "sv"'

    uncorrect_tokenizers = ["Bad String", None, 1, 1.0, [], {}]
    for uncorrect_tokenizer in uncorrect_tokenizers:
        with pytest.raises(ValueError) as exc_info:
            data_iterator = DataIterator(
                test_source_path,
                test_target_path,
                test_metadata_path,
                language_pair,
                uncorrect_tokenizer
            )
        assert str(exc_info.value) == "tokenizer must be an instance of MTTokenizer"

    tokenizer = MTTokenizer()
    with pytest.raises(Exception) as exc_info:
        data_iterator = DataIterator(
            test_source_path,
            test_target_path,
            test_metadata_path,
            language_pair,
            tokenizer
        )
    assert str(exc_info.value) == "tokenizer must be trained before using it"
    

    tokenizer.train(test_source_path, test_target_path, test_metadata_path,'"src_lang": "en", "tgt_lang": "da"' )
    with pytest.raises(Exception) as exc_info:
        data_iterator = DataIterator(
            test_source_path,
            test_target_path,
            test_metadata_path,
            language_pair,
            tokenizer
        )
    assert str(exc_info.value) == "tokenizer language pair must match the language pair filter"
    
def test_generator_language_filtering():
    test_source_path = "tests/examples/test.src"
    test_target_path = "tests/examples/test.tgt"
    test_metadata_path = "tests/examples/test.metadata"
    language_pairs = ["None", '"src_lang": "en", "tgt_lang": "sv"', '"src_lang": "nb", "tgt_lang": "da"', '"src_lang": "sv", "tgt_lang": "nb"']
    data_iterator_lengths = [1000, 83, 108, 100]

    for language_pair, data_iterator_length in zip(language_pairs, data_iterator_lengths):
        tokenizer = MTTokenizer()
        tokenizer.train(test_source_path, test_target_path, test_metadata_path, language_pair)
        data_iterator = DataIterator(
        test_source_path,
        test_target_path,
        test_metadata_path,
        language_pair,
        tokenizer
        )
        assert len(data_iterator) == data_iterator_length


def test_generator_sampling():

    test_source_path = "tests/examples/test.src"
    test_target_path = "tests/examples/test.tgt"
    test_metadata_path = "tests/examples/test.metadata"
    language_pair = '"src_lang": "en", "tgt_lang": "sv"'

    tokenizer = MTTokenizer()
    tokenizer.train(test_source_path, test_target_path, test_metadata_path, language_pair)
    data_iterator = DataIterator(
        test_source_path,
        test_target_path,
        test_metadata_path,
        language_pair,
        tokenizer
    )

    assert data_iterator.__getitem__(0)[0] == [1,2,3,4,5,6,7,5,8,9,10,11,12,13,14,15]
    assert data_iterator.__getitem__(49)[1][:15] == [107,168,185,41,168,43,412,384,413,414,415,416,417,264,418]



    
