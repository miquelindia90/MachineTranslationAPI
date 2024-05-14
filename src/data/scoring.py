import sys

import torch
from nltk.translate.bleu_score import sentence_bleu

sys.path.append("./src")

from data.tokenizer import MTTokenizer

def calculate_batch_bleu_score(output_tensor: torch.Tensor, target_tensor: torch.Tensor, target_lengths: torch.Tensor, tokenizer: MTTokenizer) -> float:
    """
    Calculates the batch BLEU score for machine translation outputs.

    Args:
        output_tensor (torch.Tensor): The output tensor from the machine translation model.
        target_tensor (torch.Tensor): The target tensor containing the ground truth translations.
        target_lengths (torch.Tensor): The lengths of the target sequences.
        tokenizer (MTTokenizer): The tokenizer used for converting word indices to words.

    Returns:
        float: The average BLEU score for the batch.

    """
    bleu_scores = []
    for i in range(output_tensor.size(0)):
        output = torch.argmax(output_tensor[i], dim=-1).squeeze()
        target, target_length = target_tensor[i,:], target_lengths[i].item()
        output = ' '.join([tokenizer.target_lang_id_to_word(word_idx.item()) for word_idx in output[:target_length-2]])
        target = ' '.join([tokenizer.target_lang_id_to_word(word_idx.item()) for word_idx in target[:target_length-2]])
        bleu_scores.append(calculate_bleu_score(output, [target]))
    return sum(bleu_scores) / len(bleu_scores)
   

def calculate_bleu_score(candidate: list, references: list) -> float:
    """
    Calculate the BLEU score between a candidate translation and a list of reference translations.
    
    Args:
        candidate (list): The candidate translation as a list of tokens.
        references (list): The list of reference translations, each as a list of tokens.
    
    Returns:
        float: The BLEU score as a percentage.
    """
    candidate_tokens = candidate.split()
    reference_tokens = [reference.split() for reference in references]
    
    # Calculate BLEU score
    bleu_score = sentence_bleu(reference_tokens, candidate_tokens)
    return round(bleu_score*100,2)
