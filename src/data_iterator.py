from data_utils import read_text_sentences, get_metadata_languages_indexes
from tokenizer import MTTokenizer

class DataIterator:
    def __init__(
        self,
        source_path: str,
        target_path: str,
        metadata_path: str,
        language_pair_filter: str = "None",
        tokenizer: MTTokenizer = MTTokenizer(),
    ):
        self._source_path = source_path
        self._target_path = target_path
        self._metadata_path = metadata_path
        self._language_pair_filter = language_pair_filter
        self._tokenizer = tokenizer
        self._validate_tokenizer()
        self.source_list, self.target_list = self.__prepare_iterator_data()

    def _validate_tokenizer(self):
        if not isinstance(self._tokenizer, MTTokenizer):
            raise ValueError("tokenizer must be an instance of MTTokenizer")
        elif self._tokenizer.get_source_tokens_dictionary() == dict() or self._tokenizer.get_target_tokens_dictionary() == dict():
            raise Exception("tokenizer must be trained before using it")
        elif self._tokenizer.get_language_pair() != self._language_pair_filter:
            raise Exception("tokenizer language pair must match the language pair filter")
        

    def __filter_lists_by_language(self, source_list: list, target_list: list) -> tuple:
        language_indexes = get_metadata_languages_indexes(
            self._metadata_path, self._language_pair_filter
        )
        indexed_source_list = [source_list[index] for index in language_indexes]
        indexed_target_list = [target_list[index] for index in language_indexes]
        return indexed_source_list, indexed_target_list

    def __prepare_iterator_data(self):
        source_list = read_text_sentences(self._source_path)
        target_list = read_text_sentences(self._target_path)
        if self._language_pair_filter != "None":
            source_list, target_list = self.__filter_lists_by_language(
                source_list, target_list
            )
        return source_list, target_list

    def __len__(self):
        return len(self.source_list)

    def __getitem__(self, index):

        return None