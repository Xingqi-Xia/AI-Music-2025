import torch
import torch.nn.functional as F
import numpy as np
from .train.models import MusicGRUVAE  # 确保 model.py 在同一目录下

class MusicEvaluator:
    def __init__(self, model_path, device=None, config=None):
        """
        初始化评估器
        :param model_path: 训练好的 .pth 文件路径
        :param device: 'cuda' 或 'cpu'
        :param config: 模型参数字典 (需与训练时一致)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 默认配置 (与之前的 train.py 保持一致)
        self.config = {
            "vocab_size": 130,
            "embed_dim": 256,
            "hidden_dim": 512,
            "latent_dim": 128,
            "seq_len": 32
        }
        if config:
            self.config.update(config)

        # 1. 初始化模型架构
        self.model = MusicGRUVAE(
            vocab_size=self.config["vocab_size"],
            embed_dim=self.config["embed_dim"],
            hidden_dim=self.config["hidden_dim"],
            latent_dim=self.config["latent_dim"],
            seq_len=self.config["seq_len"]
        ).to(self.device)

        # 2. 加载权重
        print(f"⚖️ 正在加载模型权重: {model_path} ...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 兼容处理：检查是用 save_state_dict 保存的还是整个 checkpoint
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval() #以此进入评估模式 (关闭 Dropout 等)
        
        # 目标风格向量 (初始化为空)
        self.target_centroid = None
        print("✅ 模型加载完成！")

    def set_target_style(self, target_data_tensor):
        """
        计算目标风格的中心点 (Latent Centroid)
        :param target_data_tensor: Tensor [N, 32]，包含你想要模仿的乐曲片段
        """
        print("🎯 正在计算目标风格的 Latent 向量中心...")
        target_data_tensor = target_data_tensor.to(self.device)
        
        batch_size = 512
        mus = []
        
        # 分批计算，防止显存爆炸
        with torch.no_grad():
            for i in range(0, len(target_data_tensor), batch_size):
                batch = target_data_tensor[i : i + batch_size]
                mu, _ = self.model.encode(batch)
                mus.append(mu)
        
        all_mus = torch.cat(mus, dim=0)
        
        # 计算平均向量 (Centroid)
        self.target_centroid = torch.mean(all_mus, dim=0) # [128]
        print(f"✅ 目标风格已设定。参考样本数: {len(target_data_tensor)}")

    def get_style_fitness(self, individual_seq):
        """
        核心函数：计算单个个体的适应度
        :param individual_seq: list or np.array, 长度为 32 的整数序列
        :return: float, 0.0 ~ 1.0 (越高越好)
        """
        if self.target_centroid is None:
            raise ValueError("请先调用 set_target_style() 设定目标风格！")

        # 预处理：转为 Tensor [1, 32]
        seq_tensor = torch.tensor(individual_seq, dtype=torch.long).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 1. 编码得到 latent vector
            mu, _ = self.model.encode(seq_tensor) # [1, 128]
            
            # 2. 计算与目标中心的余弦相似度 (Cosine Similarity)
            # 余弦相似度范围是 [-1, 1]
            similarity = F.cosine_similarity(mu, self.target_centroid.unsqueeze(0))
            
            # 3. 归一化到 [0, 1] 方便遗传算法使用
            # sim = 1 -> score = 1
            # sim = -1 -> score = 0
            score = (similarity.item() + 1) / 2
            
        return score

    def evaluate(self, population_grid: np.ndarray) -> np.ndarray:
        """
        批量接口，适配 GA 的 evaluator 调用。
        :param population_grid: numpy 数组，形状 [pop_size, seq_len]
        :return: numpy 数组，形状 [pop_size,]，每个个体的得分 (0~1)
        """
        if self.target_centroid is None:
            raise ValueError("请先调用 set_target_style() 设定目标风格！")

        # 转 Tensor 到模型设备，一次性编码全部个体
        pop_tensor = torch.as_tensor(population_grid, dtype=torch.long, device=self.device)
        with torch.no_grad():
            mus, _ = self.model.encode(pop_tensor)  # [B, latent_dim]
            centroid = self.target_centroid.unsqueeze(0)  # [1, latent_dim]
            similarity = F.cosine_similarity(mus, centroid)  # [B]
            scores = torch.clamp((similarity + 1) / 2, min=0.0, max=1.0)  # 归一化到 [0,1]

        return scores.detach().cpu().numpy()

    def get_playability_score(self, individual_seq):
        """
        (可选) 计算“可演奏性”或“通顺度”
        原理：如果模型重构误差很低，说明这段旋律符合训练数据的语法
        """
        seq_tensor = torch.tensor(individual_seq, dtype=torch.long).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, _, _ = self.model(seq_tensor)
            # 简单的 Loss 计算 (Negative Log Likelihood)
            # 注意：这里需要错位计算，或者直接看模型对当前序列的困惑度
            # 为简单起见，我们假设 loss 越小越好，转化成 0~1 分数
            # 这里的实现略去复杂的 loss 计算，仅作接口示意
            pass
        return 0.0

    def repair_melody(self, individual_seq):
        """
        (可选) 变异操作：修复旋律
        将旋律编码再解码，去除不协和的噪声
        """
        seq_tensor = torch.tensor(individual_seq, dtype=torch.long).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu, _ = self.model.encode(seq_tensor)
            # 使用均值解码 (不加随机噪声)
            logits = self.model.decode(mu)
            reconstructed = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return reconstructed.tolist()