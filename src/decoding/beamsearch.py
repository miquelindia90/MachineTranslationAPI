import sys
import math

import torch


class BeamSearcher:
    def __init__(
        self,
        SOS_token_id: int,
        EOS_token_id: int,
        beam_size: int = 1,
        n_best: int = 1,
        max_length: int = 150,
        length_penalty: float = 0.0,
    ):
        self.SOS_token = SOS_token_id
        self.EOS_token = EOS_token_id
        self.beam_size = beam_size
        self.n_best = n_best
        self.max_length = max_length
        self.lenghth_penalty = length_penalty
        self.reset()

    def reset(self):
        self.finished_hypotheses = list()
        self.search_is_finished = False

    def get_best_hypothesis_sequence(self):
        if not self.search_is_finished:
            print(
                "You are getting the best hypothesis before the search is finished. The best hypothesis might not be the best one."
            )
        best_sequence = sorted(
            self.finished_hypotheses, key=lambda x: x["score"], reverse=True
        )[0]["sequence"]
        return self._remove_sequence_tokens(best_sequence)

    def _remove_sequence_tokens(self, sequence: list):
        if sequence[-1] == self.EOS_token:
            sequence = sequence[:-1]
        if sequence[0] == self.SOS_token:
            sequence = sequence[1:]
        return sequence

    def _sum_sequence_scores(self, scores: list) -> float:
        total_score = 0.0
        for score in scores:
            total_score += math.log(score)
        return total_score

    def _create_step_hypothesis(
        self,
        target_tensor: torch.Tensor,
        decoder_output: torch.Tensor,
        topk_indices: torch.Tensor,
        probabilities: torch.Tensor,
    ):
        hypothesis = list()
        for batch_index in range(target_tensor.size(0)):
            initial_sequence = target_tensor[batch_index, :].tolist()
            for beam_index in range(topk_indices.size(-1)):
                sequence = initial_sequence + [
                    topk_indices[batch_index, beam_index].item()
                ]
                scores = [
                    probabilities[batch_index, sequence_index, sequence_id].item()
                    for sequence_index, sequence_id in enumerate(sequence[1:])
                ]
                accumulated_score = (
                    self._sum_sequence_scores(scores)
                    - len(sequence) * self.lenghth_penalty
                )
                hypothesis.append({"sequence": sequence, "score": accumulated_score})
        return hypothesis

    def _create_target_tensor(self, pruned_hypotheses: list):

        sequence_list = list()
        for hypothesis in pruned_hypotheses:
            sequence_list.append(hypothesis["sequence"])

        return torch.tensor(sequence_list, dtype=torch.long)

    def _check_for_finished_hypotheses(self, pruned_hypotheses: list):

        for hypothesis in pruned_hypotheses:
            if (
                hypothesis["sequence"][-1] == self.EOS_token
                or len(hypothesis["sequence"]) == self.max_length
            ):
                self.finished_hypotheses.append(hypothesis)

        for hypothesis in self.finished_hypotheses:
            if hypothesis in pruned_hypotheses:
                pruned_hypotheses.remove(hypothesis)

        return pruned_hypotheses

    def update(self, target_tensor, decoder_output):

        if self.search_is_finished:
            raise Exception(
                "The search is finished. No more updates are allowed. Reset the searcher to start a new search."
            )

        probabilities = torch.softmax(decoder_output, dim=-1)
        _, topk_indices = torch.topk(probabilities[:, -1, :], k=self.beam_size, dim=-1)
        hypothesis = self._create_step_hypothesis(
            target_tensor, decoder_output, topk_indices, probabilities
        )
        pruned_hypotheses = sorted(hypothesis, key=lambda x: x["score"], reverse=True)[
            : self.n_best
        ]
        non_finished_best_hypothesis = self._check_for_finished_hypotheses(
            pruned_hypotheses
        )
        if len(self.finished_hypotheses) >= self.n_best:
            self.search_is_finished = True
        return self._create_target_tensor(non_finished_best_hypothesis)
