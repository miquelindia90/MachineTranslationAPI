import sys
import torch

sys.path.append("./src")

from data.tokenizer import MTTokenizer
from data.scoring import calculate_bleu_score, calculate_batch_bleu_score


def test_bleu_calculation_methods():
    candidate_translation = "There is a cat in the mat"
    reference_translations = [
        "The cat is sitting on the mat",
        "There is a cat on the mat",
    ]
    assert calculate_bleu_score(candidate_translation, reference_translations) == 48.89


def test_batch_bleu_calculation():
    tokenizer = MTTokenizer()
    tokenizer.load_tokens_dictionary(
        "tests/examples/reference_source_tokens.json",
        "tests/examples/reference_target_tokens.json",
    )

    target_tensor = torch.Tensor([[0, 1, 2, 3]])
    target_lengths = torch.Tensor([4]).long()
    output_tensor = torch.Tensor(
        [
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.2, 0.3, 0.4, 0.1],
                [0.3, 0.1, 0.4, 0.2],
                [0.1, 0.3, 0.2, 0.4],
                [0.1, 0.3, 0.2, 0.4],
            ]
        ]
    )
    assert (
        calculate_batch_bleu_score(
            output_tensor, target_tensor, target_lengths, tokenizer
        )
        == 0.0
    )

    # TO DO: Add more test cases
