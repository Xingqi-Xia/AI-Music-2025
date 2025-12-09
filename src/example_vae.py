"""
用训练好的 VAE 评估器驱动遗传算法作曲的示例。

假设你已经训练好 VAE 并保存了 checkpoint（例如 checkpoints/vae_gru_bach_v1_best.pth），
并且准备了一批目标风格片段（Tensor，形状 [N, 32]），用于计算目标风格中心。

运行前请确保：
1) 已安装所需依赖（torch 等）。
2) 路径参数 MODEL_PATH / TARGET_DATA_PATH 根据实际情况更新。
"""

import os
import torch
import numpy as np

from GA import MusicGeneticOptimizer
from VAE.vae_evaluator import MusicEvaluator as VAEEvaluator
from MusicRep import MelodySequence, Synthesizer, SineStrategy


# === 配置区域：根据实际情况修改 ===
MODEL_PATH = "./VAE/train/checkpoints/vae_gru_bach_v2_latest.pth"  # 训练好的 VAE 权重
TARGET_DATA_PATH = "./VAE/train/classical_dataset.pt"             # 用于计算目标风格中心的样本
EXAMPLE_PATH = "example_outputs/vae_ga_example/"

# 可根据你的模型超参修改（需与训练时一致）
VAE_CONFIG = {
    "vocab_size": 130,
    "embed_dim": 256,
    "hidden_dim": 512,
    "latent_dim": 128,
    "seq_len": 32,
}


def main():
    os.makedirs(EXAMPLE_PATH, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 1) 准备目标风格数据，计算 latent centroid
    print(f"加载目标风格数据: {TARGET_DATA_PATH}")
    target_data = torch.load(TARGET_DATA_PATH)  # 假设返回 Tensor/ Dataset；此处直接当 Tensor
    # 若 load 返回 Dataset，可转为 Tensor: torch.stack([x for x in target_data], dim=0)

    vae_evaluator = VAEEvaluator(
        model_path=MODEL_PATH,
        device=device,
        config=VAE_CONFIG,
    )

    # 仅取前若干样本作为参考，避免过大占用内存；可根据需要调整
    max_ref = min(len(target_data), 4096)
    ref_tensor = target_data[:max_ref] if isinstance(target_data, torch.Tensor) else torch.stack([target_data[i] for i in range(max_ref)], dim=0)
    vae_evaluator.set_target_style(ref_tensor)

    # 2) 构建 GA，使用 VAE evaluator
    ga_optimizer = MusicGeneticOptimizer(
        pop_size=1000,
        n_generations=500,
        elite_ratio=0.15,
        prob_point_mutation=0.1,
        prob_transposition=0.05,
        prob_retrograde=0.02,
        prob_inversion=0.02,
        evaluator_model=vae_evaluator,
        device=device,
    )

    # 3) 初始化音频合成器（简单正弦波示例）
    synth = Synthesizer(strategy=SineStrategy())

    # 4) 初始化种群并运行 GA
    ga_optimizer._initialize_population()
    ga_optimizer.fit(verbose=True)

    # 5) 取最优个体，导出音频
    best_melody_grid = ga_optimizer.best_individual_
    best_melody = MelodySequence(best_melody_grid)

    print("最优旋律序列的音符网格:", best_melody.grid)
    output_wav = os.path.join(EXAMPLE_PATH, "best_melody.wav")
    synth.render(best_melody.grid, bpm=120, output_path=output_wav)
    print(f"已保存最优旋律的合成音频: {output_wav}")


if __name__ == "__main__":
    main()