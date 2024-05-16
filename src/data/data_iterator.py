import numpy as np
import torch

from data.data_utils import read_text_sentences, get_metadata_languages_indexes
from data.tokenizer import MTTokenizer


class DataIterator(torch.utils.data.Dataset):
    """
    A class that represents a data iterator for machine translation.

    Args:
        source_path (str): The path to the source data file.
        target_path (str): The path to the target data file.
        metadata_path (str): The path to the metadata file.
        language_pair_filter (str, optional): The language pair filter. Defaults to "None".
        tokenizer (MTTokenizer, optional): The tokenizer object. Defaults to MTTokenizer().

    Raises:
        ValueError: If the tokenizer is not an instance of MTTokenizer.
        Exception: If the tokenizer is not trained or if the tokenizer language pair does not match the language pair filter.
    """

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
        (
            self._tokenized_source_list,
            self._tokenized_target_list,
        ) = self.__prepare_iterator_data()

    def _validate_tokenizer(self) -> None:
        """
        Validates the tokenizer object.

        Raises:
            ValueError: If the tokenizer is not an instance of MTTokenizer.
            Exception: If the tokenizer is not trained or if the tokenizer language pair does not match the language pair filter.
        """
        if not isinstance(self._tokenizer, MTTokenizer):
            raise ValueError("tokenizer must be an instance of MTTokenizer")
        elif (
            self._tokenizer.get_source_tokens_dictionary() == dict()
            or self._tokenizer.get_target_tokens_dictionary() == dict()
        ):
            raise Exception("tokenizer must be trained before using it")
        elif self._tokenizer.get_language_pair() != self._language_pair_filter:
            raise Exception(
                "tokenizer language pair must match the language pair filter"
            )

    def __filter_lists_by_language(self, source_list: list, target_list: list) -> tuple:
        """
        Filters the source and target lists based on the language pair filter.

        Args:
            source_list (list): The source list of sentences.
            target_list (list): The target list of sentences.

        Returns:
            tuple: A tuple containing the filtered source list and target list.
        """
        language_indexes = get_metadata_languages_indexes(
            self._metadata_path, self._language_pair_filter
        )
        indexed_source_list = [source_list[index] for index in language_indexes]
        indexed_target_list = [target_list[index] for index in language_indexes]
        return indexed_source_list, indexed_target_list

    def _tokenize_data_sentences(self, source_list: list, target_list: list) -> tuple:
        """
        Tokenizes the source and target sentences.

        Args:
            source_list (list): The source list of sentences.
            target_list (list): The target list of sentences.

        Returns:
            tuple: A tuple containing the tokenized source list and target list.
        """
        tokenized_source_list = [
            self._tokenizer.source_lang_sentence_to_id_list(sentence)
            for sentence in source_list
        ]
        tokenized_target_list = [
            self._tokenizer.target_lang_sentence_to_id_list(sentence)
            for sentence in target_list
        ]
        return tokenized_source_list, tokenized_target_list

    def __prepare_iterator_data(self) -> tuple:
        """
        Prepares the data for the iterator.

        Returns:
            tuple: A tuple containing the tokenized source list and target list.
        """
        source_list = read_text_sentences(self._source_path)
        target_list = read_text_sentences(self._target_path)
        if self._language_pair_filter != "None":
            self.source_list, self.target_list = self.__filter_lists_by_language(
                source_list, target_list
            )
        tokenized_source_list, tokenized_target_list = self._tokenize_data_sentences(
            source_list, target_list
        )
        self.max_source_length = (
            max([len(sentence) for sentence in tokenized_source_list]) + 2
        )
        self.max_target_length = (
            max([len(sentence) for sentence in tokenized_target_list]) + 2
        )

        return tokenized_source_list, tokenized_target_list

    def get_source_sentence(self, index: int) -> str:
        return self.source_list[index]

    def get_target_sentence(self, index: int) -> str:
        return self.target_list[index]

    def __len__(self):
        return len(self._tokenized_source_list)

    def __getitem__(self, index: int) -> tuple:
        source_item_list = (
            [self._tokenizer.source_lang_word_to_id("SOS")]
            + self._tokenized_source_list[index]
            + [self._tokenizer.source_lang_word_to_id("EOS")]
        )
        target_item_list = (
            [self._tokenizer.target_lang_word_to_id("SOS")]
            + self._tokenized_target_list[index]
            + [self._tokenizer.target_lang_word_to_id("EOS")]
        )
        padded_source_item_list = source_item_list + [
            self._tokenizer.source_lang_word_to_id("PAD")
        ] * (self.max_source_length - len(source_item_list))
        padded_target_item_list = target_item_list + [
            self._tokenizer.target_lang_word_to_id("PAD")
        ] * (self.max_target_length - len(target_item_list))
        return (
            np.array(padded_source_item_list),
            np.array([len(source_item_list)]),
            np.array(padded_target_item_list),
            np.array([len(target_item_list)]),
        )
