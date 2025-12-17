# 读取pt文件，数一下总共有多少token
import torch
import numpy as np
INPUT_FILE = "giant_piano_mind_v1.pt"

def main():
    data = torch.load(INPUT_FILE)
    print(f"数据集包含 {len(data)} 条序列。")
    # 因为seq数量很大，用for循环的效率太烂了,改用numpy进行向量化
    total_tokens = np.mean(data[0:100])*len(data)   # 先算前100条的平均长度，再乘以总条数

    print(f"文件 {INPUT_FILE} 中总共有 {total_tokens} 个 token。")

if __name__ == "__main__":
    main()