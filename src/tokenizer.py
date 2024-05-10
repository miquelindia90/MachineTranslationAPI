import json

from data_utils import read_text_sentences, get_metadata_languages_indexes


class SingleTokenizer:
    """ Tokenizes sentences and maps words to unique IDs. """

    def __init__(self):
        self._tokens_dictionary = dict()

    def get_tokens_dictionary(self) -> dict:
        """ Returns the tokens dictionary. """
        return self._tokens_dictionary

    def _split_sentence_in_words(self, sentence: str) -> list:
        """ Splits a sentence into a list of lowercase words. """
        return [word.lower() for word in sentence.strip().split()]

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

    def sentence_to_id_list(self, sentence: str) -> list:
        """
        Converts a sentence to a list of word IDs.

        Args:
            sentence (str): The input sentence.

        Returns:
            A list of word IDs representing the sentence.
        """
        return self._words_to_id_list(self._split_sentence_in_words(sentence))

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


class MTTokenizer:
    def __init__(self):
        self._source_tokenizer = SingleTokenizer()
        self._target_dicitonary = SingleTokenizer()

    def get_source_tokens_dictionary(self) -> dict:
        return self._source_tokenizer.get_tokens_dictionary()

    def get_target_tokens_dictionary(self) -> dict:
        return self._target_dicitonary.get_tokens_dictionary()