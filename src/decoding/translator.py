import sys
import torch

sys.path.append("./src")

from data.tokenizer import MTTokenizer
from decoding.beamsearch import BeamSearcher


class Translator:
    def __init__(
        self,
        net: torch.nn.Module,
        tokenizer: MTTokenizer,
        device: torch.device = torch.device("cpu"),
    ):
        self.net = net
        self._init_tokenizer(tokenizer)
        self.beam_searcher = BeamSearcher(SOS_token_id=self.tokenizer.target_lang_word_to_id("SOS"), EOS_token_id=self.tokenizer.target_lang_word_to_id("EOS"))
        self.device = device
        self.net.to(self.device)

    def _init_tokenizer(self, tokenizer: MTTokenizer):
        self.tokenizer = tokenizer
        if (
            self.tokenizer.get_source_tokens_dictionary() is dict()
            or self.tokenizer.get_target_tokens_dictionary() is dict()
        ):
            raise Exception(
                "Tokenizer must be trained before using it for translation."
            )

    def _prepare_input_tensor(self, sentence: str) -> torch.Tensor:
        sentence_ids = self.tokenizer.source_lang_sentence_to_id_list(sentence)
        sentence_tensor = torch.tensor([sentence_ids], dtype=torch.long).to(self.device)
        decoder_ids = [self.tokenizer.source_lang_word_to_id("SOS")]
        decoder_tensor = torch.tensor([decoder_ids], dtype=torch.long).to(self.device)
        return sentence_tensor, decoder_tensor

    def translate(self, sentence: str) -> str:
        self.beam_searcher.reset()
        encoder_input_tensor, decoder_tensor = self._prepare_input_tensor(sentence)
        encoder_output = self.net.encoder(encoder_input_tensor, None)
        while not self.beam_searcher.search_is_finished:
            print("Decoding ...")
            decoder_output = self.net.decoder(decoder_tensor, encoder_output, None, None)
            self.beam_searcher.update(decoder_tensor, decoder_output)
        #     target_tensor = self.beam_searcher.get_current_target_tensor()
            print(decoder_output.size())
        # return self.tokenizer.target_lang_id_list_to_sentence(
        #     self.beam_searcher.get_best_hypothesis()
        # )
        return sentence
