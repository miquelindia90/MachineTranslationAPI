import sys
import pytest

sys.path.append("./src")

from data.scoring import calculate_bleu_score

def test_bleu_calculation_methods():
    candidate_translation = "There is a cat in the mat"
    reference_translations = ["The cat is sitting on the mat", "There is a cat on the mat"]
    assert calculate_bleu_score(candidate_translation, reference_translations) == 48.89

