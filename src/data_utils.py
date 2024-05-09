def _read_text_sentences(file_path: str) -> list:
    sentences = list()
    with open(file_path, "r") as input_file:
        for line in input_file:
            sentences.append(line.strip())
    return sentences


class SingleTokenizer:
    def __init__(self):
        self._tokens_dictionary = dict()

    def get_tokens_dictionary(self) -> dict:
        return self._tokens_dictionary


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
        language_pair: str = "None",
    ):
        self._source_path = source_path
        self._target_path = target_path
        self._metadata_path = metadata_path
        self._language_pair = language_pair
        self.source_list, self.target_list = self.__prepare_iterator_data()

    def __extract_language_indexes(self) -> list:
        language_indexes = list()
        languages = self._language_pair.split("-")
        with open(self._metadata_path, "r") as metadata_file:
            for index, line in enumerate(metadata_file.readlines()):
                sline = line.strip().split('"')
                if sline[3] == languages[0] and sline[7] == languages[1]:
                    language_indexes.append(index)
        return language_indexes

    def __filter_lists_by_language(self, source_list: list, target_list: list) -> tuple:
        language_indexes = self.__extract_language_indexes()
        indexed_source_list = [source_list[index] for index in language_indexes]
        indexed_target_list = [target_list[index] for index in language_indexes]
        return indexed_source_list, indexed_target_list

    def __prepare_iterator_data(self):
        source_list = _read_text_sentences(self._source_path)
        target_list = _read_text_sentences(self._target_path)
        if self._language_pair != "None":
            source_list, target_list = self.__filter_lists_by_language(
                source_list, target_list
            )
        return source_list, target_list

    def __len__(self):
        return len(self.source_list)

    def __getitem__(self, index):

        return None
