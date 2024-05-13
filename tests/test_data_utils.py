import sys

sys.path.append("./src")

from data.data_utils import read_text_sentences, get_metadata_languages_indexes



def test_read_text_sentences():
    
    file_path = "tests/examples/test.src"
    sentences = read_text_sentences(file_path)
    assert len(sentences) == 1000
    assert sentences[0] == "And it is also the case with the new national-conservative populism of today."
    assert sentences[1] == "Selv om du bruker en søkemotor, må du sørge for at den fører deg til det virkelige nettstedet."
    assert sentences[2] == "The links are maintained by their respective organisations, which are solely responsible for their content."

def test_get_metadata_languages_indexes():

    metadata_file_path = "tests/examples/test.metadata"
    language_filter_str = '"src_lang": "en"'
    indexes = get_metadata_languages_indexes(metadata_file_path, language_filter_str)
    assert len(indexes) == 202
    assert indexes[0] == 0
    assert indexes[1] == 2
    assert indexes[2] == 21

    language_filter_str = '"src_lang": "nb"'
    indexes = get_metadata_languages_indexes(metadata_file_path, language_filter_str)
    assert len(indexes) == 257
    assert indexes[0] == 1
    assert indexes[1] == 3
    assert indexes[2] == 5

    language_filter_str = '"src_lang": "sv"'
    indexes = get_metadata_languages_indexes(metadata_file_path, language_filter_str)
    assert len(indexes) == 230
    assert indexes[0] == 6
    assert indexes[1] == 7
    assert indexes[2] == 9
