import os
import glob
import numpy as np
import miditoolkit
import torch
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_DIR = "./midi_dataset_local/data"       # 刚才下载的 MIDI 文件夹
OUTPUT_FILE = "classical_dataset.pt" # 保存的 PyTorch 数据集文件
SEQ_LEN = 32                         # 序列长度 (4小节 * 8个音符/小节)
QUANTIZATION = 8                     # 1/8 音符量化
MIN_NOTE_PITCH = 21                  # 钢琴最低音 (A0)
MAX_NOTE_PITCH = 108                 # 钢琴最高音 (C8)
AUGMENT_RANGE = range(-6, 6)         # 移调增强范围
NUM_WORKERS = 32                     # CPU 并行核心数 (根据你的服务器调整)
# ===========================================

def encode_segment(notes, ticks_per_grid, segment_start_tick):
    """
    将一段 MIDI 音符列表转换为 [SEQ_LEN] 的整数向量
    编码规则: 0=Rest, 1=Hold, p+2=Pitch
    """
    # 初始化为休止符
    grid = np.zeros(SEQ_LEN, dtype=int)
    
    # 策略：Highest Pitch Priority (高音优先作为旋律)
    # 我们创建一个临时的 array 来记录每个格子的 (status, pitch)
    # status: 0=rest, 1=hold, 2=onset
    
    # 用字典记录每个 grid index 上出现的音符信息：{index: (pitch, is_onset)}
    # 如果同一格有多个音，保留 pitch 最大的
    temp_grid = {}

    segment_end_tick = segment_start_tick + SEQ_LEN * ticks_per_grid

    for note in notes:
        # 检查音符是否在这个片段的时间范围内
        if note.end < segment_start_tick or note.start >= segment_end_tick:
            continue
            
        # 计算音符在网格中的相对位置
        # Quantize start
        rel_start = max(0, note.start - segment_start_tick)
        start_idx = int(round(rel_start / ticks_per_grid))
        
        # Quantize end
        rel_end = min(segment_end_tick - segment_start_tick, note.end - segment_start_tick)
        end_idx = int(round(rel_end / ticks_per_grid))
        
        # 修正边界
        if start_idx >= SEQ_LEN: continue
        end_idx = min(end_idx, SEQ_LEN)
        if start_idx == end_idx: continue # 音符太短，忽略

        # 填充网格
        for i in range(start_idx, end_idx):
            is_onset = (i == start_idx)
            current_pitch = note.pitch
            
            # 冲突处理：保留高音
            if i not in temp_grid:
                temp_grid[i] = (current_pitch, is_onset)
            else:
                prev_pitch, prev_onset = temp_grid[i]
                if current_pitch > prev_pitch:
                    temp_grid[i] = (current_pitch, is_onset)

    # 将 temp_grid 转换为最终的 encoding
    for i in range(SEQ_LEN):
        if i in temp_grid:
            pitch, is_onset = temp_grid[i]
            if is_onset:
                grid[i] = pitch + 2 # Note On
            else:
                grid[i] = 1         # Note Hold
        else:
            grid[i] = 0             # Rest
            
    return grid

def process_one_file(file_path):
    """
    处理单个 MIDI 文件，返回多个 segments
    """
    try:
        # 加载 MIDI
        midi_obj = miditoolkit.MidiFile(file_path)
    except:
        return []

    # 1. 检查 Time Signature，如果不含 4/4 拍，或者太复杂，这里简单处理：强制按 4/4 切割
    ticks_per_beat = midi_obj.ticks_per_beat
    ticks_per_grid = ticks_per_beat / 2 # 1/8 音符 = 0.5 拍
    
    # 2. 合并所有轨道 (Flatten)
    all_notes = []
    for instrument in midi_obj.instruments:
        if instrument.is_drum: continue # 跳过鼓
        all_notes.extend(instrument.notes)
    
    if not all_notes:
        return []

    # 按时间排序
    all_notes.sort(key=lambda x: x.start)
    
    # 获取这首曲子的总时长 (ticks)
    max_tick = max(n.end for n in all_notes)
    
    # 3. 切片 (Slicing)
    segments = []
    ticks_per_segment = int(ticks_per_grid * SEQ_LEN)
    
    # 滑动窗口，步长为半个片段（Overlap 50%）以增加数据量
    stride = ticks_per_segment // 2 
    
    for start_tick in range(0, max_tick, stride):
        # 提取该窗口内的音符 (为了 encode_segment 效率，这里可以先做简单的 filter，或者把所有 notes 传进去)
        # 考虑到 encode_segment 内部有判断，直接传所有 notes 稍微慢点但逻辑简单
        # 为了性能优化，我们只传附近的 notes
        relevant_notes = [n for n in all_notes if n.end > start_tick and n.start < start_tick + ticks_per_segment]
        
        if not relevant_notes:
            continue

        # 编码
        encoded = encode_segment(relevant_notes, ticks_per_grid, start_tick)
        
        # 4. 过滤垃圾数据
        # 规则：如果全是休止符，或者音符太少（比如少于4个），丢弃
        # grid > 1 对应 pitch (>=2)
        note_count = np.sum(encoded > 1) 
        if note_count < 4: 
            continue
            
        segments.append(encoded)

    # 5. 数据增强 (Data Augmentation)
    augmented_segments = []
    for seg in segments:
        for trans in AUGMENT_RANGE:
            # 复制一份
            aug_seg = seg.copy()
            
            # 找到 pitch 部分 (value >= 2)
            pitch_mask = aug_seg >= 2
            
            # 移调
            aug_seg[pitch_mask] += trans
            
            # 检查边界
            # MIDI 范围 0-127 -> 编码范围 2-129
            # 我们限制在钢琴键范围内 21-108 -> 编码 23-110 (可选)
            # 或者只要不越界 (0-127) 即可
            valid = True
            if np.any(aug_seg[pitch_mask] < 2) or np.any(aug_seg[pitch_mask] > 129):
                valid = False
            
            if valid:
                augmented_segments.append(aug_seg)
                
    return augmented_segments

def main():
    files = glob.glob(os.path.join(INPUT_DIR, "*.mid"))
    files = files[:6000] # 如果文件太多，先测前6000个
    print(f"🎵 找到 {len(files)} 个 MIDI 文件，准备处理...")

    all_data = []
    
    # 多进程处理
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # 使用 tqdm 显示进度
        results = list(tqdm(executor.map(process_one_file, files), total=len(files)))
    
    # 汇总结果
    print("📦 正在合并数据...")
    for res in results:
        all_data.extend(res)
    
    # 转换为 PyTorch Tensor
    # 格式: (N, 32) int64
    print(f"📊 原始数据转换中... 总样本数: {len(all_data)}")
    if len(all_data) == 0:
        print("❌ 没有提取到任何数据，请检查 MIDI 文件夹路径或文件内容。")
        return

    data_tensor = torch.tensor(np.array(all_data), dtype=torch.long)
    
    # 保存
    print(f"💾 保存到 {OUTPUT_FILE} ...")
    torch.save(data_tensor, OUTPUT_FILE)
    
    print("✅ 完成！")
    print(f"最终数据集形状: {data_tensor.shape}")
    print(f"包含词表索引: {data_tensor.min()} ~ {data_tensor.max()}")

if __name__ == "__main__":
    main()