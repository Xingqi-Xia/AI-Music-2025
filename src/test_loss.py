"""
核心目标：说明transformer学会了真实的音乐。我们需要加强文章的说服力。

需要做什么：
1. 可以用很高的温度加上某个模型生成数据集。
2. 证明nano和standard模型它们的loss具有相关性。用其中一个模型生成的音乐在另一个模型上的loss也不高。
3. 纯随机变异，验证loss确实发生的升高。
4. 绘制nano percentage vs standard percentage 热力图。我们期望这个矩阵是对角占优的，相关系数趋近于1。
"""

import os
import sys
import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from transformer import GPTMusicEvaluator

model_path_standard = "./transformer/checkpoints_gpt/music_gpt_standard.pth"
model_path_nano = "./transformer/checkpoints_gpt/music_gpt_nano_datav3_best.pth"
device="cuda" if torch.cuda.is_available() else "cpu"

def generate_samples(evaluator:GPTMusicEvaluator, count=1, temperature = 1.0, progress_bar=True):
    max_new = 32-1
    top_k = 30
    results=[]
    for _ in tqdm.tqdm(range(count), disable=not progress_bar, desc="Generating samples"):
        prompt_tokens = [np.random.randint(53, 79)+2] # F3~G5
        generated = evaluator.generate(
            prompt_sequence=prompt_tokens,
            max_new_tokens=max_new,
            temperature=temperature,
            top_k=top_k,
        )
        results.append(generated)
    return np.array(results)

def batch_evaluate_loss(evaluators:list[GPTMusicEvaluator], sequences):
    """
    evaluators: list of GPTMusicEvaluator
    sequences: list of token sequences
    return: loss_matrix, shape (len(evaluators), len(sequences))
    """
    loss_matrix = np.zeros((len(evaluators), len(sequences)))
    sequences=np.array(sequences)
    for i, evaluator in tqdm.tqdm(enumerate(evaluators), total=len(evaluators), desc="Evaluating loss"):
        loss_matrix[i] = evaluator.evaluate(sequences)
    return loss_matrix

def noise_mutation(sequences, mutation_rate=0.1, mutation_range=6):
    mutate_delta= np.random.randint(-mutation_range, mutation_range+1, size=sequences.shape)
    mutation_mask = (np.random.rand(*sequences.shape) < mutation_rate) & (sequences >= 2)
    mutated=np.where(mutation_mask, np.clip(sequences + mutate_delta, 2, 128), sequences)
    return mutated

def random_sequence(count=1, length=32):
    return np.random.randint(2, 130, size=(count, length))

def main():
    evaluator_standard=GPTMusicEvaluator(model_path=model_path_standard, device=device)
    evaluator_nano=GPTMusicEvaluator(model_path=model_path_nano, device=device)
    os.makedirs("./example_outputs/test_loss", exist_ok=True)
    
    # 1. 生成或读取样本
    use_saved_data = True
    save_path = "./example_outputs/test_loss/transformer_generated_samples.npz"
    if use_saved_data and os.path.exists(save_path):
        data = np.load(save_path, allow_pickle=True)
        all_samples = data["arr_0"].item()
        print("Loaded existing samples from transformer_generated_samples.npz")
    else:
        sample_count = 100
        temperatures = np.arange(0.1, 5.1, 0.1)
        all_samples = {}
        for temp in temperatures:
            samples = generate_samples(evaluator=evaluator_standard, count=sample_count, temperature=temp)
            all_samples[temp] = samples
            print(f"Generated {sample_count} samples at temperature {temp}.")
        np.savez(save_path, all_samples)
        print("Saved generated samples to transformer_generated_samples.npz")
    
    # 2. 评估loss
    use_saved_data = True
    save_path_loss = "./example_outputs/test_loss/transformer_loss_matrix.npz"
    if use_saved_data and os.path.exists(save_path_loss):
        loss_matrix = np.load(save_path_loss)["loss_matrix"]
        print("Loss matrix already exists, skipping evaluation.")
    else:
        all_samples_concatenated = np.concatenate(list(all_samples.values()), axis=0)
        all_samples_concatenated = np.concatenate([
            all_samples_concatenated, 
            noise_mutation(all_samples_concatenated, mutation_rate=0.2), 
            noise_mutation(all_samples_concatenated, mutation_rate=0.5), 
            noise_mutation(all_samples_concatenated, mutation_rate=0.9), 
            random_sequence(len(all_samples_concatenated))
        ], axis=0)
        loss_matrix = batch_evaluate_loss(
            evaluators=[evaluator_standard, evaluator_nano],
            sequences=all_samples_concatenated)
        np.savez(save_path_loss, loss_matrix=loss_matrix)
        print("Saved loss matrix to transformer_loss_matrix.npz")
    # 3. 绘制结果
    x_min=0
    x_max=2
    plt.figure(figsize=(8, 6))
    # plt.hist2d(loss_matrix[0], loss_matrix[1], bins=100, range=[[x_min, x_max], [loss_matrix.min(), x_max]], cmap='Blues')
    # plt.colorbar(label='Counts')
    plt.scatter(loss_matrix[0], loss_matrix[1], alpha=0.5, label='Samples')
    plt.plot([loss_matrix.min(), x_max], [loss_matrix.min(), x_max], 'r--', label='y=x')
    r=np.corrcoef(loss_matrix[0], loss_matrix[1])[0, 1]
    plt.text(x_max*0.6, x_max*0.2, f"Correlation: {r:.4f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    print(f"Correlation coefficient between Standard and Nano model losses: {r:.4f}")

    plt.xlim(loss_matrix.min(), x_max)
    plt.ylim(loss_matrix.min(), x_max)
    plt.xlabel("Standard Model Loss")
    plt.ylabel("Nano Model Loss")
    plt.title("Loss Correlation between Standard and Nano Models")
    plt.legend()
    plt.grid(True)
    plt.savefig("./example_outputs/test_loss/transformer_loss_correlation.png")
    plt.show()
    

if __name__ == "__main__":
    main()
