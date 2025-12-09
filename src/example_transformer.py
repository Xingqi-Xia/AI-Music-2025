"""
Interactive CLI demo for using a trained MusicGPT (transformer) as evaluator/generator.

This version is prompt-driven (no command-line args). It will:
- Search for .pth checkpoints under ./transformer/checkpoints_gpt/ and let you pick one.
- Ask whether to run fitness scoring or melody generation.
- Accept token input in a relaxed format (spaces/commas/brackets).
- Optionally render WAV if MusicRep is available.
"""

import os
import sys
import torch
import numpy as np

# Import through package namespace to match transformer/ dependency layout
from transformer import GPTMusicEvaluator, tokens_to_melodygrid, MUSICREP_AVAILABLE

if MUSICREP_AVAILABLE:
    from MusicRep.synthesizer import Synthesizer, StringStrategy


def parse_token_string(token_str: str) -> list[int]:
    cleaned = token_str.replace("[", " ").replace("]", " ").replace(",", " ")
    parts = cleaned.strip().split()
    if not parts:
        raise ValueError("Empty token input.")
    try:
        return [int(x) for x in parts]
    except ValueError:
        raise ValueError("Tokens must be integers (separate by space or comma).")


def load_dataset_sample(dataset_path: str, index: int, max_len: int = 32) -> list[int]:
    data = torch.load(dataset_path, weights_only=False)
    if index >= len(data):
        raise IndexError(f"dataset index {index} out of range (len={len(data)})")
    sample = data[index]
    arr = sample.cpu().numpy() if isinstance(sample, torch.Tensor) else np.array(sample)
    return arr[:max_len].tolist()


def render_if_available(tokens: list[int], output_path: str, bpm: int = 100):
    if not MUSICREP_AVAILABLE:
        print("MusicRep not available, skip rendering.")
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grid = tokens_to_melodygrid(tokens)
    synth = Synthesizer(strategy=StringStrategy())
    synth.render(grid, bpm=bpm, output_path=output_path)
    print(f"✅ WAV saved to: {output_path}")


def pick_checkpoint(default_dir: str) -> str:
    if not os.path.isdir(default_dir):
        print(f"No checkpoint directory found at {default_dir}. Please place .pth files there.")
        sys.exit(1)

    pths = [f for f in os.listdir(default_dir) if f.endswith('.pth')]
    if not pths:
        print(f"No .pth files found in {default_dir}. Please add a checkpoint and rerun.")
        sys.exit(1)

    if len(pths) == 1:
        chosen = pths[0]
        print(f"Found checkpoint: {chosen}")
        return os.path.join(default_dir, chosen)

    print("Select a checkpoint:")
    for i, name in enumerate(pths):
        print(f"  [{i}] {name}")
    while True:
        sel = input("Enter index: ").strip()
        if sel.isdigit() and 0 <= int(sel) < len(pths):
            return os.path.join(default_dir, pths[int(sel)])
        print("Invalid selection, try again.")


def prompt_mode() -> str:
    while True:
        m = input("Choose mode: [f]itness / [g]enerate: ").strip().lower()
        if m in ("f", "fitness"):
            return "fitness"
        if m in ("g", "generate"):
            return "generate"
        print("Please enter 'f' or 'g'.")


def interactive_fitness(evaluator: GPTMusicEvaluator):
    use_dataset = input("Use dataset sample? (y/N): ").strip().lower() == "y"
    if use_dataset:
        dataset_path = input("Dataset path (.pt): ").strip()
        if not dataset_path:
            print("Dataset path required.")
            return
        idx_str = input("Sample index (default 0): ").strip()
        idx = int(idx_str) if idx_str.isdigit() else 0
        max_len_str = input("Max tokens to slice (default 32): ").strip()
        max_len = int(max_len_str) if max_len_str.isdigit() else 32
        seq = load_dataset_sample(dataset_path, idx, max_len)
        print(f"Loaded sample idx={idx}, len={len(seq)}")
    else:
        seq_str = input("Enter tokens (space/comma separated, e.g. 130,60,1,62,1): ").strip()
        seq = parse_token_string(seq_str)

    score = evaluator.get_fitness_score(seq)
    print(f"Fitness score: {score:.6f}")


def interactive_generate(evaluator: GPTMusicEvaluator):
    prompt_str = input("Enter prompt tokens (default BOS=130): ").strip()
    prompt_tokens = parse_token_string(prompt_str) if prompt_str else [130]
    max_new_str = input("Max new tokens (default 128): ").strip()
    max_new = int(max_new_str) if max_new_str.isdigit() else 128
    temp_str = input("Temperature (default 1.0): ").strip()
    temperature = float(temp_str) if temp_str else 1.0
    topk_str = input("Top-k (default 30): ").strip()
    top_k = int(topk_str) if topk_str.isdigit() else 30

    generated = evaluator.generate(
        prompt_sequence=prompt_tokens,
        max_new_tokens=max_new,
        temperature=temperature,
        top_k=top_k,
    )
    print(f"Generated tokens (first {min(64, len(generated))}): {generated[:64]}")

    if input("Render to WAV? (y/N): ").strip().lower() == "y":
        out_path = input("Output wav path (default example_outputs/transformer_generated/generated.wav): ").strip()
        output = out_path if out_path else "example_outputs/transformer_generated/generated.wav"
        render_if_available(generated, output)


def main():
    default_ckpt_dir = "./transformer/checkpoints_gpt"
    model_path = pick_checkpoint(default_ckpt_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    evaluator = GPTMusicEvaluator(model_path=model_path, device=device)

    mode = prompt_mode()
    if mode == "fitness":
        interactive_fitness(evaluator)
    else:
        interactive_generate(evaluator)


if __name__ == "__main__":
    main()