def read_text_sentences(file_path: str) -> list:
    """
    Read text sentences from a file.

    Args:
        file_path (str): The path to the file containing the sentences.

    Returns:
        list: A list of sentences read from the file.
    """
    sentences = list()
    with open(file_path, "r") as input_file:
        for line in input_file:
            sentences.append(line.strip())
    return sentences


def get_metadata_languages_indexes(
    metadata_file_path: str, language_filter_str: str
) -> list:
    """
    Retrieves the indexes of lines in a metadata file that contain a specific language filter string.

    Args:
        metadata_file_path (str): The path to the metadata file.
        language_filter_str (str): The language filter string to search for.

    Returns:
        list: A list of indexes corresponding to lines that contain the language filter string.
    """
    indexes = list()
    with open(metadata_file_path, "r") as metadata_file:
        for index, line in enumerate(metadata_file.readlines()):
            if language_filter_str in line.strip():
                indexes.append(index)
    return indexes
