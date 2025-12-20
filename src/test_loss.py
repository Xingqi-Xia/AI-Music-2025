"""
核心目标：说明transformer学会了真实的音乐。我们需要加强文章的说服力。

需要做什么：
1. 可以用很高的温度加上某个模型生成数据集。
2. 证明nano和standard模型它们的fitness具有相关性。用其中一个模型生成的音乐在另一个模型上的fitness也不高。
3. 纯随机变异，验证fitness确实发生的升高。
4. 绘制nano percentage vs standard percentage 热力图。我们期望这个矩阵是对角占优的，相关系数趋近于1。
"""

import os
import sys
import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from transformer import GPTMusicEvaluator
from scipy.stats import linregress

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

def batch_evaluate_fitness(evaluators:list[GPTMusicEvaluator], sequences):
    """
    evaluators: list of GPTMusicEvaluator
    sequences: list of token sequences
    return: fitness_matrix, shape (len(evaluators), len(sequences))
    """
    fitness_matrix = np.zeros((len(evaluators), len(sequences)))
    sequences=np.array(sequences)
    for i, evaluator in tqdm.tqdm(enumerate(evaluators), total=len(evaluators), desc="Evaluating fitness"):
        fitness_matrix[i] = evaluator.evaluate(sequences)
    return fitness_matrix

def noise_mutation(sequences, mutation_rate=0.1, mutation_range=6):
    mutate_delta= np.random.randint(-mutation_range, mutation_range+1, size=sequences.shape)
    mutation_mask = (np.random.rand(*sequences.shape) < mutation_rate) & (sequences >= 2)
    mutated=np.where(mutation_mask, np.clip(sequences + mutate_delta, 2, 128), sequences)
    return mutated

def random_sequence(count=1, length=32):
    return np.random.randint(2, 130, size=(count, length))

def main():
    save_path = "./example_outputs/test_loss/transformer_generated_samples.npz"
    save_path_fitness = "./example_outputs/test_loss/transformer_loss_matrix.npz"
    if not os.path.exists(model_path_standard) or not os.path.exists(model_path_nano):
        evaluator_standard=GPTMusicEvaluator(model_path=model_path_standard, device=device)
        evaluator_nano=GPTMusicEvaluator(model_path=model_path_nano, device=device)
    os.makedirs("./example_outputs/test_loss", exist_ok=True)
    
    # 1. 生成或读取样本
    if os.path.exists(save_path):
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
    
    # 2. 评估fitness
    if os.path.exists(save_path_fitness):
        fitness_matrix = np.load(save_path_fitness)["loss_matrix"]
        print("fitness matrix already exists, skipping evaluation.")
    else:
        all_samples_concatenated = np.concatenate(list(all_samples.values()), axis=0)
        all_samples_concatenated = np.concatenate([
            all_samples_concatenated, 
            noise_mutation(all_samples_concatenated, mutation_rate=0.2), 
            noise_mutation(all_samples_concatenated, mutation_rate=0.5), 
            noise_mutation(all_samples_concatenated, mutation_rate=0.9), 
            random_sequence(len(all_samples_concatenated))
        ], axis=0)
        fitness_matrix = batch_evaluate_fitness(
            evaluators=[evaluator_standard, evaluator_nano],
            sequences=all_samples_concatenated)
        np.savez(save_path_fitness, loss_matrix=fitness_matrix)
        print("Saved fitness matrix to transformer_loss_matrix.npz")
    # 3. 绘制Nano与Standard模型fitness的相关性图
    x_min=0
    x_max=3
    plt.figure(figsize=(8, 6))
    # plt.hist2d(fitness_matrix[0], fitness_matrix[1], bins=100, range=[[x_min, x_max], [fitness_matrix.min(), x_max]], cmap='Blues')
    # plt.colorbar(label='Counts')
    plt.scatter(fitness_matrix[0], fitness_matrix[1], alpha=0.5, label='Samples')
    k, b, r, p, std_err = linreg_result = linregress(fitness_matrix[0], fitness_matrix[1])
    plt.plot([x_min, x_max], [k*x_min + b, k*x_max + b], color='red', label=f'Fit line: y={k:.2f}x+{b:.2f}')
    plt.text(x_max*0.6, x_max*0.2, f"Correlation: {r:.4f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    print(f"Correlation coefficient between Standard and Nano model fitnesses: {r:.4f}")

    plt.xlim(x_min, x_max)
    plt.ylim(x_min, x_max)
    plt.xlabel("Standard Model fitness")
    plt.ylabel("Nano Model fitness")
    plt.title("fitness Correlation between Standard and Nano Models")
    plt.legend()
    plt.grid(True)
    plt.savefig("./example_outputs/test_loss/transformer_loss_correlation.png")
    plt.show()
    plt.close()

    # 绘制分位数热力图
    # 获取每个数据的分位数
    percentage_standards = np.array([np.mean(fitness_matrix[0] <= x) for x in fitness_matrix[0]])
    percentage_nanos = np.array([np.mean(fitness_matrix[1] <= x) for x in fitness_matrix[1]])
    plt.figure(figsize=(8, 6))
    plt.hist2d(percentage_standards, percentage_nanos, bins=10, range=[[0, 1], [0, 1]], cmap='Blues')
    # 向对角线和次对角线上每一个格子内写上该格子内数据总数的百分比
    for i in range(10):
        for j in range(10):
            if abs(i - j) > 1:
                continue
            count=np.sum(
                (percentage_standards >= i/10) & (percentage_standards < (i+1)/10) &
                (percentage_nanos >= j/10) & (percentage_nanos < (j+1)/10)
            )
            plt.text(
                i/10 + 0.05, j/10 + 0.05, 
                f"{count/len(percentage_standards)*100:.1f}%", 
                color='white' if i==j else 'black', 
                fontsize=12, ha='center', va='center'
            )

    # r= np.corrcoef(percentage_standards, percentage_nanos)[0,1]
    # plt.text(0.6,0.2, f"Correlation: {r:.4f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.colorbar(label='Counts')
    plt.xlabel("Standard Model fitness percentage")
    plt.ylabel("Nano Model fitness percentage")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("fitness Percentage Correlation between Standard and Nano Models")
    plt.tight_layout()
    plt.savefig("./example_outputs/test_loss/transformer_loss_heatmap.png")
    plt.show()
    plt.close()

    # 4. 绘制两个模型的fitness随温度变化图
    all_samples_temperatures = np.concatenate([[temp]*sample.shape[0] for temp,sample in all_samples.items()])
    all_samples_temperatures = np.concatenate([all_samples_temperatures, all_samples_temperatures, all_samples_temperatures, all_samples_temperatures, [np.inf]*len(all_samples_temperatures)])
    fitnesses_at_temperatures = {temp: fitness_matrix[:,all_samples_temperatures == temp] for temp in all_samples.keys() if temp<=2}
    fitnesses_at_temperatures[np.inf] = fitness_matrix[:, np.isinf(all_samples_temperatures)]
    mean_fitnesses_at_temperatures = {temp: fitnesses.mean(axis=1) for temp, fitnesses in fitnesses_at_temperatures.items()}

    temps = sorted(mean_fitnesses_at_temperatures.keys())
    standard_means = [mean_fitnesses_at_temperatures[temp][0] for temp in temps]
    nano_means = [mean_fitnesses_at_temperatures[temp][1] for temp in temps]
    standard_means_std = [fitnesses_at_temperatures[temp][0].std()/np.sqrt(len(fitnesses_at_temperatures[temp][0])-1) for temp in temps]
    nano_means_std = [fitnesses_at_temperatures[temp][1].std()/np.sqrt(len(fitnesses_at_temperatures[temp][1])-1) for temp in temps]

    # 绘制柱状图
    plt.figure(figsize=(12, 6))
    bar_width=0.4
    x_positions=np.arange(len(temps))
    colors=['#1f77b4', '#ff7f0e']
    bars1 = plt.bar(x_positions - bar_width/2, standard_means, width=bar_width, color=colors[0], label='Standard Model')
    bars2 = plt.bar(x_positions + bar_width/2, nano_means, width=bar_width, color=colors[1], label='Nano Model')
    for bar in bars1 + bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=5)
    plt.xlabel('TEMPERATURE')
    plt.ylabel('Average fitness')
    plt.title('Average fitness vs Temperature for Standard and Nano Models')
    plt.xticks(x_positions, [f"Temp={temp:.1f}" if np.isfinite(temp) else "random" for temp in temps], rotation=45)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("./example_outputs/test_loss/transformer_loss_vs_temperature.png")
    plt.show()
    plt.close()

    # 绘制带有误差的折线图
    plt.figure(figsize=(12, 6))
    x_positions=np.arange(len(temps))
    plt.errorbar(
        x_positions,
        standard_means,
        yerr=standard_means_std,
        label='Standard Model',
        capsize=5,
        fmt='-+'
    )
    plt.errorbar(
        x_positions,
        nano_means,
        yerr=nano_means_std,
        label='Nano Model',
        capsize=5,
        fmt='-x'
    )
    plt.xlabel('TEMPERATURE')
    plt.ylabel('Average fitness')
    plt.title('Average fitness vs Temperature for Standard and Nano Models')
    plt.xticks(x_positions, [f"Temp={temp:.1f}" if np.isfinite(temp) else "random" for temp in temps], rotation=45)
    plt.xlim(-1, len(temps))
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("./example_outputs/test_loss/transformer_loss_vs_temperature_with_errorbar.png")
    plt.show()
    plt.close()
    

if __name__ == "__main__":
    main()
