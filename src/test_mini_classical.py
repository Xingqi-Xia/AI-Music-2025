# 从数据集(dataset.pt)读取一个列，然后把它用方法转换为wav，验证预处理是否正常

DATASET_PATH="./transformer/dataset/classical_gpt_dataset_smart.pt"

import torch
import numpy as np
from MusicRep import MelodySequence, Synthesizer, StringStrategy
import os

# 读取数据集（预处理脚本保存的是 list/ndarray 序列，而非 state_dict）
# torch 2.6+ 默认 weights_only=True，会阻止非张量对象；这里显式关闭，并允许 numpy reconstruct
try:
    torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
except Exception:
    pass

dataset = torch.load(DATASET_PATH, weights_only=False)
print(f"数据集包含 {len(dataset)} 个样本。")

# 随机选择一个样本进行测试
sample = dataset[1145]  # 这里选择第一个样本，你也可以随机选择
print(sample)

# 兼容 Tensor / ndarray / list
if isinstance(sample, torch.Tensor):
    melody_grid = sample.cpu().numpy()[:128]
else:
    melody_grid = np.array(sample)[:128]

print("选取的旋律网格:", melody_grid)

# 使用 MusicRep 的 MelodySequence 和 Synthesizer 播放
#melody = MelodySequence(melody_grid)
synth = Synthesizer(strategy=StringStrategy())
OUTPUT_FOLDER="example_outputs/test_mini_classical/"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
output_wav_path = os.path.join(OUTPUT_FOLDER, "test_mini_classical.wav")
synth.render(melody_grid, bpm=120, output_path=output_wav_path)
print(f"已保存合成音频为 {output_wav_path}")