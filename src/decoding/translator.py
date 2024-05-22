import sys
import torch

sys.path.append("./src")

from data.tokenizer import MTTokenizer
from decoding.beamsearch import BeamSearcher


class Translator:
    """
    A class that represents a machine translation translator.

    Args:
        net (torch.nn.Module): The neural network model used for translation.
        tokenizer (MTTokenizer): The tokenizer used for tokenizing the input and output sentences.
        device (torch.device, optional): The device on which the model will be run (default: torch.device("cpu")).

    Attributes:
        net (torch.nn.Module): The neural network model used for translation.
        tokenizer (MTTokenizer): The tokenizer used for tokenizing the input and output sentences.
        beam_searcher (BeamSearcher): The beam searcher used for decoding the output sequence.
        device (torch.device): The device on which the model will be run.

    """

    def __init__(
        self,
        net: torch.nn.Module,
        tokenizer: MTTokenizer,
        device: torch.device = torch.device("cpu"),
    ):
        self.net = net
        self._init_tokenizer(tokenizer)
        self.beam_searcher = BeamSearcher(
            SOS_token_id=self.tokenizer.target_lang_word_to_id("SOS"),
            EOS_token_id=self.tokenizer.target_lang_word_to_id("EOS"),
        )
        self.device = device
        self.net.to(self.device)

    def _init_tokenizer(self, tokenizer: MTTokenizer):
        """
        Initializes the tokenizer.

        Args:
            tokenizer (MTTokenizer): The tokenizer used for tokenizing the input and output sentences.

        Raises:
            Exception: If the tokenizer is not trained.

        """
        self.tokenizer = tokenizer
        if (
            self.tokenizer.get_source_tokens_dictionary() is dict()
            or self.tokenizer.get_target_tokens_dictionary() is dict()
        ):
            raise Exception(
                "Tokenizer must be trained before using it for translation."
            )

    def _prepare_input_tensor(self, sentence: str) -> torch.Tensor:
        """
        Prepares the input sentence for translation.

        Args:
            sentence (str): The input sentence to be translated.

        Returns:
            torch.Tensor: The input sentence tensor.

        """
        sentence_ids = self.tokenizer.source_lang_sentence_to_id_list(sentence)
        sentence_tensor = torch.tensor([sentence_ids], dtype=torch.long).to(self.device)
        decoder_ids = [self.tokenizer.source_lang_word_to_id("SOS")]
        decoder_tensor = torch.tensor([decoder_ids], dtype=torch.long).to(self.device)
        return sentence_tensor, decoder_tensor

    def translate(self, sentence: str) -> str:
        """
        Translates the input sentence.

        Args:
            sentence (str): The input sentence to be translated.

        Returns:
            str: The translated sentence.

        """
        self.beam_searcher.reset()
        encoder_input_tensor, decoder_tensor = self._prepare_input_tensor(sentence)
        encoder_output = self.net.encoder(encoder_input_tensor, None)
        while not self.beam_searcher.search_is_finished:
            decoder_output = self.net.decoder(
                decoder_tensor, encoder_output, None, None
            )
            decoder_tensor = self.beam_searcher.update(decoder_tensor, decoder_output)
            decoder_tensor = decoder_tensor.to(self.device)
        raw_sentence = " ".join(
            self.tokenizer.target_lang_id_to_word(word_id)
            for word_id in self.beam_searcher.get_best_hypothesis_sequence()
        )
        return raw_sentence
