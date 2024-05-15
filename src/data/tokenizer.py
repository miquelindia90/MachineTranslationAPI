import re
import json

from data.data_utils import read_text_sentences, get_metadata_languages_indexes


def split_alphanumeric(sentence: str):
    """
    Splits not alphanumeric characters from words in a sentence.

    Args:
        sentence (str): The input sentence to be split.

    Returns:
        str: The fixed sentence with alphabetical/non-alphabetical words separated by spaces.
    """
    # TO DO: Fix this regex, is a super truck of expressions.

    new_words = re.split(
        r"(?<=\w)(?=[.,'\"*+\-%~·#¿?!])|(?<=[.,'\"*+\-%~·#¿?!])(?=\w)|(?<=[.,'\"*+\-%~·#¿?!])(?=[.,'\"*+\-%~·#¿?!])",
        sentence,
    )
    return " ".join(new_words)


class SingleTokenizer:
    """Tokenizes sentences and maps words to unique IDs."""

    def __init__(self):
        self._tokens_dictionary = dict()

    def get_tokens_dictionary(self) -> dict:
        """Returns the tokens dictionary."""
        return self._tokens_dictionary

    def _split_sentence_in_words(self, sentence: str) -> list:
        """Splits a sentence into a list of lowercase words."""
        return [word.lower() for word in split_alphanumeric(sentence).split()]

    def save_tokens_dictionary(self, output_file_path: str) -> None:
        """
        Save the tokens dictionary to a JSON file.

        Args:
            output_file_path (str): The path to the output JSON file.
        """

        with open(output_file_path, "w") as output_file:
            json.dump(self._tokens_dictionary, output_file, indent=4)

    def load_tokens_dictionary(self, json_file_path: str) -> None:
        """
        Loads the tokens dictionary from a JSON file.

        Args:
            json_file_path (str): The path to the input JSON file.
        """
        with open(json_file_path, "r") as input_file:
            self._tokens_dictionary = json.load(input_file)

    def word_to_id(self, word: str) -> int:
        """
        Maps a word to its corresponding ID in the tokens dictionary.

        Args:
            word (str): The input word.

        Returns:
            The ID of the word if it exists in the tokens dictionary, otherwise the ID of the "UNK" token.
        """
        if word in self._tokens_dictionary:
            return self._tokens_dictionary[word]
        else:
            return self._tokens_dictionary["UNK"]

    def id_to_word(self, id: int) -> str:
        """
        Maps an ID to its corresponding word in the tokens dictionary.

        Args:
            id (int): The input ID.

        Returns:
            The word corresponding to the ID.
        """
        for word, word_id in self._tokens_dictionary.items():
            if word_id == id:
                return word
        return "UNK"

    def sentence_to_id_list(self, sentence: str) -> list:
        """
        Converts a sentence to a list of word IDs.

        Args:
            sentence (str): The input sentence.

        Returns:
            A list of word IDs representing the sentence.
        """
        return self._words_to_id_list(self._split_sentence_in_words(sentence))

    def list_id_to_word_list(self, sentence: str) -> list:
        """
        Converts a list of word ids to a list of words.

        Args:
            sentence (str): The input sentence.

        Returns:
            A list of words representing the sentence.
        """
        return [self.id_to_word(int(word_id)) for word_id in sentence.split()]

    def _words_to_id_list(self, words: list) -> list:
        """
        Converts a list of words to a list of word IDs.

        Args:
            word (list): The input list of words.

        Returns:
            A list of word IDs representing the input words.
        """
        return [self.word_to_id(word) for word in words]

    def train(
        self,
        file_path: str,
        metadata_path: str = "None",
        language_filter_str: str = '"src_lang": "da"',
    ) -> None:
        """
        Trains the model using the provided file path and optional metadata path and language filter.

        Args:
            file_path (str): The path to the file containing the training data.
            metadata_path (str, optional): The path to the metadata file. Defaults to "None".
            language_filter_str (str, optional): The language filter string. Defaults to '"src_lang": "da"'.

        Returns:
            None
        """
        self._tokens_dictionary["SOS"] = 0
        sentences = read_text_sentences(file_path)
        if metadata_path != "None":
            sentences = [
                sentences[index]
                for index in get_metadata_languages_indexes(
                    metadata_path, language_filter_str
                )
            ]
        for sentence in sentences:
            for word in self._split_sentence_in_words(sentence):
                if word not in self._tokens_dictionary:
                    self._tokens_dictionary[word] = len(self._tokens_dictionary)
        self._tokens_dictionary["UNK"] = len(self._tokens_dictionary)
        self._tokens_dictionary["EOS"] = len(self._tokens_dictionary)
        self._tokens_dictionary["PAD"] = len(self._tokens_dictionary)


class MTTokenizer:
    """
    The MTTokenizer class is responsible for tokenizing source and target language sentences,
    creating and managing tokens dictionaries, and providing methods for converting words and sentences
    to their corresponding token IDs.
    """

    def __init__(self):
        self.language_pair = "None"
        self._source_tokenizer = SingleTokenizer()
        self._target_dicitonary = SingleTokenizer()

    def get_language_pair(self) -> str:
        """Get the language pair used for tokenization."""
        return self.language_pair

    def get_source_tokens_dictionary(self) -> dict:
        """Get the tokens dictionary for the source language."""
        return self._source_tokenizer.get_tokens_dictionary()

    def get_target_tokens_dictionary(self) -> dict:
        """Get the tokens dictionary for the target language."""
        return self._target_dicitonary.get_tokens_dictionary()

    def save_tokens_dictionary(
        self, source_output_file_path: str, target_output_file_path: str
    ) -> None:
        """
        Save the tokens dictionaries to the specified output file paths.

        Args:
            source_output_file_path (str): The file path to save the source tokens dictionary.
            target_output_file_path (str): The file path to save the target tokens dictionary.
        """
        self._source_tokenizer.save_tokens_dictionary(source_output_file_path)
        self._target_dicitonary.save_tokens_dictionary(target_output_file_path)

    def load_tokens_dictionary(
        self, source_file_path: str, target_file_path: str
    ) -> None:
        """
        Load the tokens dictionaries from the specified file paths.

        Args:
            source_file_path (str): The file path to load the source tokens dictionary from.
            target_file_path (str): The file path to load the target tokens dictionary from.
        """
        self._source_tokenizer.load_tokens_dictionary(source_file_path)
        self._target_dicitonary.load_tokens_dictionary(target_file_path)

    def source_lang_word_to_id(self, word: str) -> int:
        """
        Convert a word in the source language to its corresponding token ID.

        Args:
            word (str): The word to convert.

        Returns:
            int: The token ID of the word.
        """
        return self._source_tokenizer.word_to_id(word)

    def source_lang_sentence_to_id_list(self, sentence: str) -> list:
        """
        Convert a sentence in the source language to a list of token IDs.

        Args:
            sentence (str): The sentence to convert.

        Returns:
            list: A list of token IDs representing the sentence.
        """
        return self._source_tokenizer.sentence_to_id_list(sentence)

    def target_lang_word_to_id(self, word: str) -> int:
        """
        Convert a word in the target language to its corresponding token ID.

        Args:
            word (str): The word to convert.

        Returns:
            int: The token ID of the word.
        """
        return self._target_dicitonary.word_to_id(word)

    def target_lang_id_to_word(self, id: int) -> str:
        """
        Convert a token ID in the target language to its corresponding word.

        Args:
            id (int): The token ID to convert.

        Returns:
            str: The word corresponding to the token ID.
        """
        return self._target_dicitonary.id_to_word(id)

    def target_lang_sentence_to_id_list(self, sentence: str) -> list:
        """
        Convert a sentence in the target language to a list of token IDs.

        Args:
            sentence (str): The sentence to convert.

        Returns:
            list: A list of token IDs representing the sentence.
        """
        return self._target_dicitonary.sentence_to_id_list(sentence)

    def target_lang_list_id_to_word_list(self, id_list: list) -> list:
        """
        Convert a list of token IDs in the target language to a list of words.

        Args:
            sentence (str): The sentence to convert.

        Returns:
            list: A list of words representing the sentence.
        """
        return self._target_dicitonary.list_id_to_word_list(id_list)

    def train(
        self,
        src_file_path: str,
        tgt_file_path: str,
        metadata_path: str = "None",
        language_filter_str: str = '"src_lang": "en", "tgt_lang": "sv"',
    ) -> None:
        """
        Train the tokenizer using the specified source and target language files.

        Args:
            src_file_path (str): The file path of the source language file.
            tgt_file_path (str): The file path of the target language file.
            metadata_path (str, optional): The file path of the metadata file. Defaults to "None".
            language_filter_str (str, optional): The language filter string. Defaults to '"src_lang": "en", "tgt_lang": "sv"'.
        """
        self.language_pair = language_filter_str
        self._source_tokenizer.train(src_file_path, metadata_path, language_filter_str)
        self._target_dicitonary.train(tgt_file_path, metadata_path, language_filter_str)
