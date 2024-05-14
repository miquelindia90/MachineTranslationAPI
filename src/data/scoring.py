import nltk
from nltk.translate.bleu_score import sentence_bleu

def calculate_bleu_score(candidate, references):
    candidate_tokens = candidate.split()
    reference_tokens = [reference.split() for reference in references]
    
    # Calculate BLEU score
    bleu_score = sentence_bleu(reference_tokens, candidate_tokens)
    return round(bleu_score*100,2)
