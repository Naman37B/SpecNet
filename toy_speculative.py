from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch


@dataclass(frozen=True)
class ToyModel:
    """A tiny deterministic language model over a fixed vocabulary."""

    name: str
    transition_logits: torch.Tensor

    def next_logits(self, last_token: int) -> torch.Tensor:
        return self.transition_logits[last_token]

    def sample_tokens(self, prompt: Sequence[int], count: int) -> List[int]:
        tokens = list(prompt)
        generated: List[int] = []
        for _ in range(count):
            logits = self.next_logits(tokens[-1])
            token = int(torch.argmax(logits).item())
            tokens.append(token)
            generated.append(token)
        return generated


def softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def acceptance_probability(target_probs: torch.Tensor, draft_probs: torch.Tensor, token_id: int) -> float:
    draft_prob = float(draft_probs[token_id].item())
    if draft_prob == 0.0:
        return 1.0
    return min(1.0, float(target_probs[token_id].item()) / draft_prob)


def speculative_decode(
    draft_model: ToyModel,
    target_model: ToyModel,
    prompt: Sequence[int],
    draft_count: int = 5,
    max_steps: int = 5,
) -> List[int]:
    context = list(prompt)
    accepted: List[int] = []

    for _ in range(max_steps):
        draft_tokens = draft_model.sample_tokens(context, draft_count)
        accepted_this_round: List[int] = []

        for candidate in draft_tokens:
            draft_logits = draft_model.next_logits(context[-1])
            target_logits = target_model.next_logits(context[-1])
            draft_probs = softmax(draft_logits)
            target_probs = softmax(target_logits)

            probability = acceptance_probability(target_probs, draft_probs, candidate)
            if probability >= 1.0 or torch.rand(()) < probability:
                context.append(candidate)
                accepted.append(candidate)
                accepted_this_round.append(candidate)
            else:
                replacement = int(torch.multinomial(target_probs, num_samples=1).item())
                context.append(replacement)
                accepted.append(replacement)
                accepted_this_round.append(replacement)
                break

        if len(accepted_this_round) < draft_count:
            break

    return accepted


def build_toy_models(vocab_size: int = 8) -> tuple[ToyModel, ToyModel]:
    draft_logits = torch.tensor(
        [
            [6.0, 5.0, 1.0, 0.5, 0.1, -0.5, -1.0, -2.0],
            [0.2, 4.8, 5.2, 0.4, 0.2, -0.1, -0.5, -1.5],
            [0.1, 0.2, 0.3, 4.0, 4.4, 0.5, 0.0, -0.8],
            [0.0, -0.2, 0.1, 0.3, 0.2, 5.4, 5.8, 0.4],
            [0.3, 0.1, -0.1, 0.0, 0.5, 0.8, 5.0, 5.4],
            [5.2, 0.4, 0.2, 0.1, -0.2, 0.0, -0.5, -1.0],
            [0.2, 0.1, 0.4, 5.1, 5.7, 0.3, -0.1, -0.8],
            [0.0, 0.3, 0.2, 0.1, 0.0, 5.5, 5.9, 0.2],
        ],
        dtype=torch.float32,
    )
    target_logits = torch.tensor(
        [
            [6.2, 4.3, 1.2, 0.7, 0.2, -0.1, -0.7, -2.2],
            [0.0, 5.1, 5.6, 0.2, 0.1, -0.3, -0.8, -1.8],
            [0.0, 0.0, 0.4, 4.6, 5.0, 0.2, -0.1, -0.7],
            [-0.1, -0.3, 0.0, 0.4, 0.3, 5.9, 6.3, 0.1],
            [0.1, 0.0, -0.2, -0.1, 0.3, 0.7, 5.4, 5.8],
            [5.6, 0.2, 0.1, 0.0, -0.4, -0.1, -0.6, -1.2],
            [0.1, 0.0, 0.2, 5.4, 6.0, 0.1, -0.2, -0.9],
            [-0.1, 0.2, 0.1, 0.0, -0.1, 5.8, 6.2, 0.0],
        ],
        dtype=torch.float32,
    )

    if draft_logits.shape != target_logits.shape:
        raise ValueError("Toy models must share the same vocabulary shape.")
    if draft_logits.shape[0] != vocab_size:
        raise ValueError("Toy model tables do not match the requested vocab size.")

    return ToyModel("draft", draft_logits), ToyModel("target", target_logits)


def main() -> None:
    torch.manual_seed(7)

    draft_model, target_model = build_toy_models()
    prompt = [0]
    generated = speculative_decode(draft_model, target_model, prompt, draft_count=5, max_steps=4)

    print("prompt:", prompt)
    print("generated tokens:", generated)
    print("generated length:", len(generated))


if __name__ == "__main__":
    main()
