import os
import shutil
from datasets import load_dataset
from tqdm import tqdm

# =================配置区域=================
# 1. 设置 Hugging Face 镜像站 (必须在 import datasets 之前或刚开始设置)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 数据集名称
DATASET_NAME = "drengskapur/midi-classical-music"

# 3. 本地保存路径
# 用于保存 huggingface 的 dataset 格式（方便以后直接 load_from_disk）
HF_DISK_PATH = "./midi_dataset_hf"
# 用于保存解压出来的 .mid 文件（供 music21/miditoolkit 读取）
MIDI_OUTPUT_DIR = "./raw_midi_files"
# =========================================

def download_and_process():
    print(f"🚀 正在通过镜像站 ({os.environ['HF_ENDPOINT']}) 下载数据集...")
    
    # 下载数据集
    # split="train" 表示直接下载训练集部分，通常这个数据集只有 train
    ds = load_dataset(DATASET_NAME, split="train")
    
    print(f"✅ 下载完成！数据集包含 {len(ds)} 首乐曲。")
    print(ds)

    # 保存 Hugging Face 原生格式到本地（作为备份）
    print(f"💾 正在保存 Dataset 对象到 {HF_DISK_PATH} ...")
    ds.save_to_disk(HF_DISK_PATH)
    
    # 准备提取 MIDI 文件
    if not os.path.exists(MIDI_OUTPUT_DIR):
        os.makedirs(MIDI_OUTPUT_DIR)
        
    print(f"📂 正在将 MIDI 二进制文件提取到 {MIDI_OUTPUT_DIR} ...")
    
    # 遍历数据集并写入文件
    # 该数据集的结构通常包含 'genre', 'composer', 'title', 'midi_content' 等字段
    # 我们需要确认存储二进制数据的列名，通常这个数据集里二进制数据可能在 'midi' 或 'content' 列
    # 让我们先动态检测一下列名
    column_names = ds.column_names
    print(f"ℹ️  数据列名: {column_names}")
    
    # 假设二进制数据在 'midi' 列 (如果是其他列名代码会自动调整，这里做个简单的查找逻辑)
    # 对于 drengskapur/midi-classical-music，通常只有一列或者直接包含 content
    # 如果数据集中没有直接的文件名，我们用索引命名
    
    success_count = 0
    
    for idx, item in tqdm(enumerate(ds), total=len(ds)):
        try:
            # 获取 MIDI 二进制数据
            # 不同的 dataset 结构不同，这里针对通用情况做处理
            # 经查阅该数据集，通常只有一列，内容可能就是 binary 或者是 url
            # 如果是 binary (bytes)，直接写；如果是 dict 包含 'bytes'，取出来
            
            midi_data = None
            
            # 尝试常见的键名
            keys_to_check = ['midi', 'content', 'file', 'data']
            for k in keys_to_check:
                if k in item:
                    midi_data = item[k]
                    break
            
            # 如果没找到键名，且只有一列，直接取第一列
            if midi_data is None and len(item.values()) == 1:
                midi_data = list(item.values())[0]

            if midi_data is None:
                continue

            # 构建文件名
            # 尽量使用 composer 和 title，如果没有则用 ID
            composer = item.get('composer', 'unknown').replace('/', '_').strip()
            title = item.get('title', str(idx)).replace('/', '_').strip()
            filename = f"{idx}_{composer}_{title}.mid"
            
            # 限制文件名长度，防止 Linux 报错
            if len(filename) > 200:
                filename = f"{idx}.mid"
                
            filepath = os.path.join(MIDI_OUTPUT_DIR, filename)

            # 写入文件
            with open(filepath, "wb") as f:
                if isinstance(midi_data, dict) and 'bytes' in midi_data:
                    f.write(midi_data['bytes']) # 有些 HF dataset 会把 binary 放在 {'bytes': ...}
                elif isinstance(midi_data, bytes):
                    f.write(midi_data)
                else:
                    # 如果是其他格式，可能需要特殊处理，但通常是 bytes
                    pass
            
            success_count += 1
            
        except Exception as e:
            print(f"❌ 提取第 {idx} 个文件失败: {e}")

    print(f"✨ 处理完成！成功提取 {success_count} 个 MIDI 文件到 {MIDI_OUTPUT_DIR}")

if __name__ == "__main__":
    download_and_process()