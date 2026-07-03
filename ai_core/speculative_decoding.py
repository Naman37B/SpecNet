from __future__ import annotations

import math
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_models():
    print("Initializing models and tokenizer...")
    model_name_draft = "facebook/opt-125m"
    model_name_target = "facebook/opt-350m"

    # Both models MUST share the exact same tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_target)

    # Detect GPU or fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    draft_model = AutoModelForCausalLM.from_pretrained(model_name_draft)
    target_model = AutoModelForCausalLM.from_pretrained(model_name_target)

    draft_model.eval()
    target_model.eval()

    draft_model.to(device)
    target_model.to(device)

    return draft_model, target_model, tokenizer, device


@torch.no_grad()
def speculative_decoding(
    draft_model,
    target_model,
    tokenizer,
    prompt: str,
    max_len: int = 30,
    K: int = 5,
    temperature: float = 1.0,
) -> str:
    device = next(draft_model.parameters()).device

    # Tokenize input prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prefix_len = input_ids.shape[1]

    print(f"\nStarting speculative decoding loop (K={K} lookahead tokens), max_len={max_len}...")

    # For metrics
    total_drafted = 0
    total_accepted = 0

    # Continue until desired generation length reached
    while input_ids.shape[1] < prefix_len + max_len:
        current_len = input_ids.shape[1]

        # --- PHASE 1: DRAFTING ---
        draft_ids = input_ids.clone()
        draft_probs_list: List[tuple[int, float]] = []
        draft_tokens: List[int] = []

        for _ in range(K):
            outputs = draft_model(draft_ids)
            next_token_logits = outputs.logits[:, -1, :] / (temperature or 1.0)
            next_token_probs = F.softmax(next_token_logits, dim=-1)

            next_token = torch.multinomial(next_token_probs, num_samples=1)
            token_id = int(next_token[0, 0].item())
            draft_tokens.append(token_id)

            chosen_prob = float(next_token_probs[0, token_id].item())
            draft_probs_list.append((token_id, chosen_prob))

            draft_ids = torch.cat([draft_ids, next_token], dim=-1)

        total_drafted += len(draft_tokens)

        # --- PHASE 2 & 3: TARGET VERIFICATION + REJECTION SAMPLING ---
        accepted_this_round: List[int] = []

        for i, (draft_token, p_draft) in enumerate(draft_probs_list):
            # For correctness we compute the target distribution given the current prefix
            outputs_t = target_model(input_ids)
            target_next_logits = outputs_t.logits[:, -1, :] / (temperature or 1.0)
            target_next_probs = F.softmax(target_next_logits, dim=-1)

            p_target = float(target_next_probs[0, draft_token].item())

            alpha = min(1.0, p_target / (p_draft + 1e-12))

            if torch.rand(1).item() < alpha:
                # accept drafted token
                accepted_token = draft_token
            else:
                # sample replacement from target
                replacement = torch.multinomial(target_next_probs, num_samples=1)
                accepted_token = int(replacement[0, 0].item())

            # append accepted token to prefix
            input_ids = torch.cat([input_ids, torch.tensor([[accepted_token]], device=device)], dim=-1)
            accepted_this_round.append(accepted_token)

            total_accepted += 1

            # If we rejected a draft token (i.e., replacement), stop consuming further draft tokens
            if accepted_token != draft_token:
                break

            # otherwise continue to next drafted token

        # If we accepted fewer than K tokens, stop outer loop (similar to spec algorithm)
        if len(accepted_this_round) < K:
            break

    # Decode generated continuation (excluding prompt)
    generated_ids = input_ids[0, prefix_len :].tolist()
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Print simple metrics
    acceptance_rate = total_accepted / max(1, total_drafted)
    print(f"Generated {len(generated_ids)} tokens; acceptance rate: {acceptance_rate:.3f}")

    return decoded


def main():
    draft_model, target_model, tokenizer, device = load_models()
    prompt = "The future of distributed AI is"
    out = speculative_decoding(draft_model, target_model, tokenizer, prompt, max_len=30, K=5)
    print("\nPrompt:", prompt)
    print("Decoded continuation:", out)


if __name__ == "__main__":
    main()
