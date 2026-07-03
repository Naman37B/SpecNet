from __future__ import annotations

import argparse
from typing import List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def acceptance_probability(target_probs: torch.Tensor, draft_probs: torch.Tensor, token_id: int) -> float:
    draft_prob = float(draft_probs[token_id].item())
    if draft_prob == 0.0:
        return 1.0
    return min(1.0, float(target_probs[token_id].item()) / draft_prob)


def next_token_logits(model: AutoModelForCausalLM, input_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        outputs = model(input_ids.to(device))
        # logits shape: (batch, seq_len, vocab)
        logits = outputs.logits
        return logits[0, -1].cpu()


def speculative_decode_hf(
    draft_model: AutoModelForCausalLM,
    target_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_ids: List[int],
    draft_count: int = 5,
    max_steps: int = 5,
    device: torch.device = torch.device("cpu"),
) -> List[int]:
    prefix = list(prompt_ids)
    accepted: List[int] = []

    for _ in range(max_steps):
        # Build a draft block autoregressively from the draft model
        draft_block: List[int] = []
        draft_context = list(prefix)
        for _ in range(draft_count):
            input_ids = torch.tensor([draft_context], dtype=torch.long)
            draft_logits = next_token_logits(draft_model, input_ids, device)
            draft_token = int(torch.argmax(draft_logits).item())
            draft_block.append(draft_token)
            draft_context.append(draft_token)

        accepted_this_round: List[int] = []

        # Verify sequentially with the target model using the growing prefix
        for candidate in draft_block:
            input_ids = torch.tensor([prefix], dtype=torch.long)
            draft_logits = next_token_logits(draft_model, input_ids, device)
            target_logits = next_token_logits(target_model, input_ids, device)

            draft_probs = softmax(draft_logits)
            target_probs = softmax(target_logits)

            prob = acceptance_probability(target_probs, draft_probs, candidate)
            if prob >= 1.0 or torch.rand(()) < prob:
                prefix.append(candidate)
                accepted.append(candidate)
                accepted_this_round.append(candidate)
            else:
                # replacement sampled from target
                replacement = int(torch.multinomial(target_probs, num_samples=1).item())
                prefix.append(replacement)
                accepted.append(replacement)
                accepted_this_round.append(replacement)
                break

        if len(accepted_this_round) < draft_count:
            break

    return accepted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", default="facebook/opt-125m")
    parser.add_argument("--target", default="facebook/opt-350m")
    parser.add_argument("--prompt", default="The future of distributed AI is")
    parser.add_argument("--draft_count", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading tokenizer and models (this may download weights)...")
    tokenizer = AutoTokenizer.from_pretrained(args.draft)
    # ensure tokenizer is shared
    draft_model = AutoModelForCausalLM.from_pretrained(args.draft)
    target_model = AutoModelForCausalLM.from_pretrained(args.target)

    draft_model.eval()
    target_model.eval()

    if device.type == "cuda":
        draft_model.to(device)
        target_model.to(device)

    prompt_ids = tokenizer(args.prompt, return_tensors="pt").input_ids[0].tolist()

    generated_ids = speculative_decode_hf(
        draft_model,
        target_model,
        tokenizer,
        prompt_ids,
        draft_count=args.draft_count,
        max_steps=args.max_steps,
        device=device,
    )

    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print("prompt:", args.prompt)
    print("generated ids:", generated_ids)
    print("decoded:", decoded)


if __name__ == "__main__":
    main()
