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
INPUT_DIR = "./data"
OUTPUT_FILE = "classical_gpt_dataset_smart_v2.pt" # Changed filename for version 2

# 音乐参数
QUANTIZATION = 8         # 1/8 音符颗粒度
MIN_PITCH = 21
MAX_PITCH = 108
AUGMENT_RANGE = range(-5, 7) 

# 过滤参数
MIN_SEQ_LEN = 64         
MAX_REST_RATIO = 0.8    

NUM_WORKERS = 16

# --- ✨ 新增: 特殊 Token 定义 ---
# 你的音乐 Token ID 是 0 (Rest), 1 (Hold), 2-129 (Pitch)
# 我们从 130 开始定义特殊 Token
BOS_TOKEN = 130  # Begin of Song
EOS_TOKEN = 131  # End of Song
# 你的新 vocab_size 将是 132 (0-131)
# ==============================================

# 🕵️‍♂️ 关键词库 (保持不变)
MELODY_KEYWORDS = [
    'melody', 'vocal', 'voice', 'sing', 'lead', 'solo', 'theme', 'main', 
    'soprano', 'upper', 'top'
]
INSTRUMENT_KEYWORDS = [
    'violin', 'flute', 'oboe', 'clarinet', 'trumpet', 'piccolo', 
    'right', 'r.h.', 'rh' # 钢琴右手
]
ACCOMP_KEYWORDS = [
    'bass', 'drum', 'perc', 'pedal', 'left', 'l.h.', 'lh', 'lower', 
    'bottom', 'accomp', 'back', 'bg', 'strum', 'chord'
]

def score_track(instrument, total_ticks):
    """
    给单个轨道打分，判断它是否为主旋律 (保持不变)
    """
    if instrument.is_drum:
        return -9999

    score = 0
    name = instrument.name.lower().strip()
    
    for kw in MELODY_KEYWORDS:
        if kw in name: score += 100
    for kw in INSTRUMENT_KEYWORDS:
        if kw in name: score += 50
    for kw in ACCOMP_KEYWORDS:
        if kw in name: score -= 100

    notes = instrument.notes
    if len(notes) < 10:
        return -9999

    pitches = [n.pitch for n in notes]
    avg_pitch = np.mean(pitches)
    std_pitch = np.std(pitches)

    if avg_pitch > 60: score += 20
    elif avg_pitch < 45: score -= 20
    if std_pitch < 2: score -= 50

    density = len(notes) / (total_ticks / 480)
    if density > 8: score -= 10

    return score

def extract_smart_melody(midi_obj):
    """
    智能选取最佳轨道，并在该轨道上应用 Skyline 提取单声部 (保持不变)
    """
    ticks_per_beat = midi_obj.ticks_per_beat
    ticks_per_grid = ticks_per_beat / 2
    max_tick = midi_obj.max_tick
    total_grids = int(math.ceil(max_tick / ticks_per_grid))

    best_track = None
    best_score = -float('inf')

    valid_instruments = [i for i in midi_obj.instruments if not i.is_drum]
    if len(valid_instruments) == 1:
        best_track = valid_instruments[0]
    else:
        for instrument in valid_instruments:
            score = score_track(instrument, max_tick)
            if score > best_score:
                best_score = score
                best_track = instrument
    
    if best_track is None:
        return None

    melody_grid = np.zeros(total_grids, dtype=np.int16)
    pitch_grid = np.full(total_grids, -1, dtype=np.int16)
    
    notes = best_track.notes
    notes.sort(key=lambda x: x.start)

    for note in notes:
        start_idx = int(round(note.start / ticks_per_grid))
        end_idx = int(round(note.end / ticks_per_grid))
        
        if start_idx >= total_grids: continue
        end_idx = min(end_idx, total_grids)
        if start_idx == end_idx: continue

        note_pitch = note.pitch

        for i in range(start_idx, end_idx):
            if note_pitch > pitch_grid[i]:
                pitch_grid[i] = note_pitch
                if i == start_idx:
                    melody_grid[i] = note_pitch + 2
                else:
                    melody_grid[i] = 1

    return melody_grid

def process_one_file(file_path):
    """
    主要修改在这里：在数据增强后添加 BOS/EOS tokens
    """
    try:
        midi_obj = miditoolkit.MidiFile(file_path)
    except:
        return []

    raw_seq = extract_smart_melody(midi_obj)
    
    if raw_seq is None or len(raw_seq) < MIN_SEQ_LEN:
        return []

    rest_count = np.sum(raw_seq == 0)
    if rest_count / len(raw_seq) > MAX_REST_RATIO:
        return []

    pitch_indices = np.where(raw_seq >= 2)[0]
    if len(pitch_indices) == 0: return []
    original_pitches = raw_seq[pitch_indices]

    augmented_seqs_with_tokens = []
    for semitone in AUGMENT_RANGE:
        # 1. 先进行移调
        new_seq = raw_seq.copy()
        new_pitches = original_pitches + semitone
        min_token = MIN_PITCH + 2
        max_token = MAX_PITCH + 2
        
        if np.any(new_pitches < min_token) or np.any(new_pitches > max_token):
            continue
            
        new_seq[pitch_indices] = new_pitches
        
        # --- ✨ 核心修改 ---
        # 2. 在移调后的纯音乐序列前后添加 BOS 和 EOS
        final_seq = np.concatenate(
            [
                np.array([BOS_TOKEN], dtype=np.int16),
                new_seq,
                np.array([EOS_TOKEN], dtype=np.int16)
            ]
        )
        augmented_seqs_with_tokens.append(final_seq)

    return augmented_seqs_with_tokens

def main():
    files = glob.glob(os.path.join(INPUT_DIR, "*.mid"))
    
    print(f"🕵️‍♂️ 使用【智能轨道评分策略 + BOS/EOS Tokens】处理 {len(files)} 个 MIDI 文件...")
    
    all_sequences = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(executor.map(process_one_file, files), total=len(files)))

    for res in results:
        if res:
            all_sequences.extend(res)
    
    print(f"📊 提取完成！总有效序列数: {len(all_sequences)}")
    if len(all_sequences) > 0:
        lengths = [len(s) for s in all_sequences]
        print(f"   平均长度: {np.mean(lengths):.2f} (包含 BOS/EOS)")
        print(f"   总 Token: {np.sum(lengths) / 1e6:.2f} M")
        
        print(f"💾 正在保存到 {OUTPUT_FILE} ...")
        torch.save(all_sequences, OUTPUT_FILE)
    else:
        print("❌ 未提取到数据，请检查过滤条件。")

if __name__ == "__main__":
    main()