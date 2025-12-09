import torch
import torch.nn.functional as F
import numpy as np
import random
import os

from .model import MusicGPT, GPT_CONFIG

# 采用绝对路径导入 MusicRep（如果可用）
MUSICREP_AVAILABLE = False
try:
    from MusicRep.melody_sqeuence import MelodySequence
    from MusicRep.synthesizer import Synthesizer, StringStrategy
    MUSICREP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: 'MusicRep' library not found ({e}). WAV rendering will be disabled.")
    MUSICREP_AVAILABLE = False

class GPTMusicEvaluator:
    """Evaluator wrapper for trained MusicGPT: fitness + generation."""

    def __init__(self, model_path: str, device: str | None = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"⚖️ Using device: {self.device}")

        # Load checkpoint
        print(f"   Loading checkpoint from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        # Recover config
        self.config = checkpoint.get("config", {})
        if not self.config:
            print("Warning: Model config not found in checkpoint. Using default GPT_CONFIG.")
            self.config["model_config"] = GPT_CONFIG

        # Ensure vocab size for BOS/EOS
        self.config["model_config"]["vocab_size"] = 132

        # Build model
        self.model = MusicGPT(self.config["model_config"])

        # Load weights (strip DataParallel prefixes if any)
        state_dict = checkpoint["model_state_dict"]
        unwrapped_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(unwrapped_state_dict)

        self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded and ready.")

    # ---------- Fitness ----------
    def evaluate(self, population_grid: np.ndarray) -> np.ndarray:
        """Batch fitness: higher is better. population_grid shape [B, T]."""
        scores = []
        with torch.no_grad():
            for seq in population_grid:
                seq_tensor = torch.tensor(seq, dtype=torch.long, device=self.device).unsqueeze(0)
                if seq_tensor.size(1) < 2:
                    scores.append(0.0)
                    continue
                X = seq_tensor[:, :-1]
                Y = seq_tensor[:, 1:]
                _, loss = self.model(X, targets=Y)
                scores.append(1.0 / (loss.item() + 1e-6))
        return np.array(scores, dtype=np.float32)

    # ---------- Single fitness helper ----------
    def get_fitness_score(self, sequence: list[int]) -> float:
        return float(self.evaluate(np.array([sequence]))[0])

    # ---------- Generation ----------
    def generate(self, prompt_sequence: list[int], max_new_tokens: int, temperature: float = 0.8, top_k: int = 20) -> list[int]:
        prompt_tensor = torch.tensor(prompt_sequence, dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            generated_tensor = self.model.generate(
                prompt_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        return generated_tensor.squeeze(0).tolist()


# ---------- Utility: tokens -> MusicRep grid ----------
def tokens_to_melodygrid(tokens):
    grid = np.array(tokens)
    pitches = grid[grid >= 2]
    grid[grid >= 2] = pitches - 2
    return grid


if __name__ == "__main__":
    MODEL_PATH = "./transformer/checkpoints_gpt/music_gpt_v1_best.pth"
    DATA_PATH = "./transformer/dataset/classical_gpt_dataset_smart_v2.pt"
    BOS_TOKEN = 130
    OUTPUT_FOLDER = "example_outputs/transformer_generated/"

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}. Please check the path.")

    evaluator = GPTMusicEvaluator(model_path=MODEL_PATH)

    print("\n" + "=" * 50)
    print("--- 1. Testing Fitness Evaluation ---")
    print("=" * 50)

    dataset = torch.load(DATA_PATH, weights_only=False)
    good_sequence = dataset[10][:32]
    bad_sequence = [random.randint(0, 129) for _ in range(32)]
    boring_sequence = [60, 1, 62, 1, 64, 1, 62, 1] * 4

    good_fitness = evaluator.get_fitness_score(list(good_sequence))
    bad_fitness = evaluator.get_fitness_score(bad_sequence)
    boring_fitness = evaluator.get_fitness_score(boring_sequence)

    print(f"Fitness of a REAL music snippet:   {good_fitness:.4f}")
    print(f"Fitness of a BORING sequence:     {boring_fitness:.4f}")
    print(f"Fitness of RANDOM NOISE:          {bad_fitness:.4f}")

    print("\n" + "=" * 50)
    print("--- 2. Testing Melody Generation ---")
    print("=" * 50)

    prompt_from_scratch = [BOS_TOKEN]
    conservative_melody = evaluator.generate(
        prompt_sequence=prompt_from_scratch,
        max_new_tokens=128,
        temperature=1.5,
        top_k=20,
    )
    print(f"Generated sequence (first 32 tokens): {conservative_melody[1:33]}")

    c_major_prompt = [BOS_TOKEN, 60 + 2, 64 + 2, 67 + 2]
    creative_melody = evaluator.generate(
        prompt_sequence=c_major_prompt,
        max_new_tokens=128,
        temperature=1.5,
        top_k=50,
    )
    print(f"Generated sequence (first 32 tokens): {creative_melody[0:128]}")

    if MUSICREP_AVAILABLE:
        print("\n" + "=" * 50)
        print("--- 3. Rendering to WAV files ---")
        print("=" * 50)

        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        synth = Synthesizer(strategy=StringStrategy())

        conservative_grid = tokens_to_melodygrid(conservative_melody)
        conservative_path = os.path.join(OUTPUT_FOLDER, "conservative_melody.wav")
        synth.render(conservative_grid, bpm=100, output_path=conservative_path)
        print(f"✅ Conservative melody saved to: {conservative_path}")

        creative_grid = tokens_to_melodygrid(creative_melody)
        creative_path = os.path.join(OUTPUT_FOLDER, "creative_melody.wav")
        synth.render(creative_grid, bpm=100, output_path=creative_path)
        print(f"✅ Creative melody saved to: {creative_path}")