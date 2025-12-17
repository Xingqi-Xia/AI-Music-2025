
# 读取一个.pt文件, 计算其中包含的token数量
import torch
def count_tokens_in_pt_file(file_path):
    data = torch.load(file_path)
    print(data.shape[0])
    # 数据是一堆序列，每个序列包含若干token, 但用for循环遍历太慢了
    return data.numel()
    # return sum(len(seq) for seq in data)
if __name__ == '__main__':
    file_path = './dataset/giant_piano_mind_v1.pt'
    num_tokens = count_tokens_in_pt_file(file_path)
    print(f'Total number of tokens in {file_path}: {num_tokens}')