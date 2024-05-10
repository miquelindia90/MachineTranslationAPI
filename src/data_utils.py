import json


def _read_text_sentences(file_path: str) -> list:
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


def _get_metadata_languages_indexes(
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


class SingleTokenizer:
    def __init__(self):
        self._tokens_dictionary = dict()

    def get_tokens_dictionary(self) -> dict:
        return self._tokens_dictionary

    def _split_sentence_in_words(self, sentence: str) -> list:
        return [word.lower() for word in sentence.strip().split()]

    def save_tokens_dictionary(self, output_file_path: str):
        with open(output_file_path, "w") as output_file:
            json.dump(self._tokens_dictionary, output_file, indent=4)

    def load_tokens_dictionary(self, json_file_path: str):
        with open(json_file_path, "r") as input_file:
            self._tokens_dictionary = json.load(input_file)


    def word_to_id(self, word: str) -> int:
        if word in self._tokens_dictionary:
            return self._tokens_dictionary[word]
        else:
            return self._tokens_dictionary["UNK"]
        
    def sentence_to_id_list(self, sentence: str) -> list:
        return self._words_to_id_list(self._split_sentence_in_words(sentence))
        
    def _words_to_id_list(self, words: list) -> list:
        return [self.word_to_id(word) for word in words]

    def train(
        self,
        file_path: str,
        metadata_path: str = "None",
        language_filter_str: str = '"src_lang": "da"',
    ) -> None:
        self._tokens_dictionary["SOS"] = 0
        sentences = _read_text_sentences(file_path)
        if metadata_path != "None":
            sentences = [
                sentences[index]
                for index in _get_metadata_languages_indexes(
                    metadata_path, language_filter_str
                )
            ]
        for sentence in sentences:
            for word in self._split_sentence_in_words(sentence):
                if word not in self._tokens_dictionary:
                    self._tokens_dictionary[word] = len(self._tokens_dictionary)
        self._tokens_dictionary["UNK"] = len(self._tokens_dictionary)
        self._tokens_dictionary["EOS"] = len(self._tokens_dictionary)


class MTTokenizer:
    def __init__(self):
        self._source_tokenizer = SingleTokenizer()
        self._target_dicitonary = SingleTokenizer()

    def get_source_tokens_dictionary(self) -> dict:
        return self._source_tokenizer.get_tokens_dictionary()

    def get_target_tokens_dictionary(self) -> dict:
        return self._target_dicitonary.get_tokens_dictionary()


class DataIterator:
    def __init__(
        self,
        source_path: str,
        target_path: str,
        metadata_path: str,
        language_pair_filter: str = "None",
    ):
        self._source_path = source_path
        self._target_path = target_path
        self._metadata_path = metadata_path
        self._language_pair_filter = language_pair_filter
        self.source_list, self.target_list = self.__prepare_iterator_data()

    def __filter_lists_by_language(self, source_list: list, target_list: list) -> tuple:
        language_indexes = _get_metadata_languages_indexes(
            self._metadata_path, self._language_pair_filter
        )
        indexed_source_list = [source_list[index] for index in language_indexes]
        indexed_target_list = [target_list[index] for index in language_indexes]
        return indexed_source_list, indexed_target_list

    def __prepare_iterator_data(self):
        source_list = _read_text_sentences(self._source_path)
        target_list = _read_text_sentences(self._target_path)
        if self._language_pair_filter != "None":
            source_list, target_list = self.__filter_lists_by_language(
                source_list, target_list
            )
        return source_list, target_list

    def __len__(self):
        return len(self.source_list)

    def __getitem__(self, index):

        return None
