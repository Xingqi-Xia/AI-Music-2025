import os
import glob
import math
import numpy as np
import miditoolkit
import torch
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import re

# ================= ⚙️ 配置区域 =================
# --- 输入/输出路径 ---
INPUT_DIR = "./gp_dataset"  # 存放你的 MIDI 文件的数据集目录
OUTPUT_FILE = "giant_piano_mind_v1.pt" # 预处理后输出的数据集文件名

# --- 音乐核心参数 ---
# 量化精度，8 代表以八分音符为最小单位
QUANTIZATION = 8         
# 模型能处理的最低和最高 MIDI 音高 (钢琴的音域是 21-108)
MIN_PITCH = 21
MAX_PITCH = 108
# 数据增强：移调范围。-5代表向下移5个半音，6代表向上移6个半音
AUGMENT_RANGE = range(-5, 7) 

# --- 数据过滤参数 ---
# 序列最短长度（以量化单位计），太短的音乐片段可能没有意义
MIN_SEQ_LEN = 64         
# 最大休止比例，休止符太多的序列可能不是好的旋律
MAX_REST_RATIO = 0.8    

# --- 系统参数 ---
# 使用的 CPU 核心数，建议根据你的机器配置调整
NUM_WORKERS = 24

# --- ✨ 特殊 Token 定义 ---
# 0: Rest (休止)
# 1: Hold (延长)
# 2 ~ 129: Pitch (音高 0 ~ 127)
# ---
# 接下来定义特殊Token
BOS_TOKEN = 130  # Begin of Song (曲子开始)
EOS_TOKEN = 131 # End of Song (曲子结束)
# 你的 vocab_size 将是 132 (0-131)
# ==============================================

def read_single_midi(file_path):
    """
    读取单个 MIDI 文件，进行初步量化，并返回排序后的音符对象列表。
    """
    try:
        midi_obj = miditoolkit.MidiFile(file_path)
    except Exception:
        # 在多进程中，打印过多信息会混乱，返回 None 即可
        return None

    # 将所有轨道的音符合并到一个列表中进行处理
    all_notes = []
    for instrument in midi_obj.instruments:
        all_notes.extend(instrument.notes)

    if not all_notes:
        return None

    # 量化音符的开始和结束时间
    ticks_per_quantization_step = midi_obj.ticks_per_beat / QUANTIZATION
    for note in all_notes:
        note.start = int(round(note.start / ticks_per_quantization_step))
        note.end = int(round(note.end / ticks_per_quantization_step))

    # 按开始时间排序，如果开始时间相同，则按音高降序排序（优先处理高音）
    all_notes.sort(key=lambda n: (n.start, -n.pitch))

    return all_notes

def notes_to_sequence(notes):
    """
    将音符对象列表转换为模型可用的 token 序列。
    """
    if not notes:
        return None

    # 计算序列的总长度
    # 如果 note.end 为 0 会导致序列长度为 0，所以至少为 1
    last_end_time = max(note.end for note in notes) if notes else 0
    if last_end_time == 0:
        return None
    
    # 创建一个以-1为占位符的时间网格
    sequence = -1 * np.ones(last_end_time, dtype=int)

    for note in notes:
        # [FIX] 增加边界检查，忽略开始时间超出序列长度的无效音符
        if note.start >= last_end_time:
            continue
            
        # 忽略超出预设音域范围的音符
        if not (MIN_PITCH <= note.pitch <= MAX_PITCH):
            continue

        # 确保音符的结束时间不超出序列长度
        end_time = min(note.end, last_end_time)

        # 在音符起始位置，如果该位置为空，则填入音高 token
        if sequence[note.start] == -1:
            sequence[note.start] = note.pitch - MIN_PITCH + 2 # 映射到 Token ID

        # 在音符持续期间，如果该位置为空，则填入 "Hold" token
        for i in range(note.start + 1, end_time):
            if sequence[i] == -1:
                sequence[i] = 1  # Hold token

    # 将所有剩余的 -1（即无任何音符活动的时间点）替换为 "Rest" token
    sequence[sequence == -1] = 0 # Rest token
    
    final_sequence = sequence.tolist()

    # --- 数据过滤 ---
    # 过滤掉太短的序列
    if len(final_sequence) < MIN_SEQ_LEN:
        return None
    # 过滤掉休止符比例过高的序列
    if final_sequence.count(0) / len(final_sequence) > MAX_REST_RATIO:
        return None

    return final_sequence

def augment_and_finalize(sequence):
    """
    对单个序列进行数据增强（移调）并添加BOS/EOS标记。
    """
    augmented_sequences = []
    # Pitch token 的有效范围
    min_pitch_token = 2
    max_pitch_token = MAX_PITCH - MIN_PITCH + 2

    for semitone in AUGMENT_RANGE:
        # semitone=0 代表原始序列
        if semitone == 0:
            final_seq = [BOS_TOKEN] + sequence + [EOS_TOKEN]
            augmented_sequences.append(final_seq)
            continue
        
        # --- 执行移调 ---
        transposed_seq = []
        is_valid = True
        for token in sequence:
            # 只对音高 token 进行移调
            if token >= min_pitch_token: 
                new_token = token + semitone
                # 检查移调后的音高是否在有效范围内
                if not (min_pitch_token <= new_token <= max_pitch_token):
                    is_valid = False
                    break
                transposed_seq.append(new_token)
            else:
                # Rest (0) 和 Hold (1) token 保持不变
                transposed_seq.append(token)
        
        if is_valid:
            final_seq = [BOS_TOKEN] + transposed_seq + [EOS_TOKEN]
            augmented_sequences.append(final_seq)
            
    return augmented_sequences

def process_single_file(file_path):
    """
    处理单个MIDI文件的完整流程: 读取 -> 转换 -> 增强。
    这是每个子进程执行的任务。
    """
    notes = read_single_midi(file_path)
    if not notes:
        return []

    sequence = notes_to_sequence(notes)
    if not sequence:
        return []

    return augment_and_finalize(sequence)

def main():
    """
    主函数：查找所有MIDI文件，并行处理并保存结果。
    """
    # 查找所有 .mid 或 .midi 后缀的文件
    midi_files = glob.glob(os.path.join(INPUT_DIR, '**', '*.mid'), recursive=True) + \
                 glob.glob(os.path.join(INPUT_DIR, '**', '*.midi'), recursive=True)
    
    print(f"找到 {len(midi_files)} 个 MIDI 文件。开始处理...")

    all_sequences = []
    # 使用 ProcessPoolExecutor 进行并行处理
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # 使用 tqdm 创建进度条
        results = list(tqdm(executor.map(process_single_file, midi_files), total=len(midi_files)))

    # 将所有子进程返回的结果（序列列表的列表）合并成一个大列表
    for seq_list in results:
        all_sequences.extend(seq_list)

    print(f"\n处理完成！共生成 {len(all_sequences)} 条序列。")
    print("正在保存数据集...")
    # 将所有序列数据保存为 PyTorch Tensor 文件
    if all_sequences:
        torch.save(all_sequences, OUTPUT_FILE)
        print(f"数据集已成功保存到: {OUTPUT_FILE}")
    else:
        print("警告：没有生成任何有效的序列，未创建输出文件。")

if __name__ == '__main__':
    main()