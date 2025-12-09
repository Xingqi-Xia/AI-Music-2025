import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BaseVAE(nn.Module):
    """
    VAE 基类，包含重参数化技巧
    """
    def __init__(self):
        super(BaseVAE, self).__init__()

    def reparameterize(self, mu, logvar):
        """
        Reparameterization Trick: z = mu + epsilon * sigma
        训练时引入随机性，推理时直接返回 mu
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

# ============================================================================
# 模型一：基于 GRU 的 VAE (推荐)
# 优点：Latent Space 连续性好，不易发生后验坍塌，适合做风格距离度量
# ============================================================================

class MusicGRUVAE(BaseVAE):
    def __init__(self, vocab_size=130, embed_dim=256, hidden_dim=512, latent_dim=128, seq_len=32):
        super(MusicGRUVAE, self).__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # ------ Encoder ------
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 双向 GRU 提取上下文特征
        self.encoder_gru = nn.GRU(embed_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        
        # 将双向 GRU 最后的 hidden state 映射到均值和方差
        # bidirectional=True, hidden_dim * 2
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim * 2, latent_dim)

        # ------ Decoder ------
        # 将 Latent Vector 映射回 GRU 的初始 Hidden State
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        
        self.decoder_gru = nn.GRU(embed_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        self.dropout = nn.Dropout(0.2)

    def encode(self, x):
        """
        输入序列 x -> (mu, logvar)
        """
        embedded = self.embedding(x) # [B, Seq, Emb]
        _, h_n = self.encoder_gru(embedded) # h_n: [Num_Layers*2, B, Hidden]
        
        # 取最后一层的 hidden state，拼接正向和反向
        # h_n 的结构是 (layer1_fwd, layer1_bwd, layer2_fwd, layer2_bwd)
        # 我们取最后两层拼接
        h_last = torch.cat((h_n[-2], h_n[-1]), dim=1) # [B, Hidden*2]
        
        mu = self.fc_mu(h_last)
        logvar = self.fc_var(h_last)
        return mu, logvar

    def decode(self, z, x_input=None):
        """
        z: Latent Vector
        x_input: Teacher Forcing 的输入。如果为 None，则进行自回归生成（Inference模式）
        """
        batch_size = z.size(0)
        
        # 初始化 Decoder 的 hidden state
        h_0 = self.decoder_input(z) # [B, Hidden]
        # GRU 需要 num_layers 维度的 hidden state
        # 这里简单地复制 hidden state 给每一层 (Layer=2)
        h_0 = h_0.unsqueeze(0).repeat(2, 1, 1) # [2, B, Hidden]

        if x_input is not None:
            # Teacher Forcing 分支：训练和验证都应保持同一逻辑，避免 eval 时走自回归导致形状不匹配
            embedded = self.embedding(x_input)
            embedded = self.dropout(embedded)
            output, _ = self.decoder_gru(embedded, h_0)
            logits = self.fc_out(output)
            return logits

        # === 推理模式 (Autoregressive Generation) ===
        # 这是一个慢速循环，但在 GA 变异时可能用到
        generated = []
        # 初始输入可以是全 0 (Rest) 或者特定的 Start Token
        # 假设我们用 0 (Rest) 开始，或者需要外部传入第一个音
        # 这里简单处理：第一个音设为 0 (Rest)
        curr_input = torch.zeros(batch_size, 1, dtype=torch.long, device=z.device)
        curr_hidden = h_0
        
        for _ in range(self.seq_len):
            embed = self.embedding(curr_input) # [B, 1, Emb]
            output, curr_hidden = self.decoder_gru(embed, curr_hidden)
            logit = self.fc_out(output) # [B, 1, Vocab]
            
            # 贪婪采样 (Greedy Search) 或 随机采样
            # 这里用贪婪采样
            next_token = torch.argmax(logit, dim=-1)
            generated.append(next_token)
            curr_input = next_token
        
        return torch.cat(generated, dim=1)

    def forward(self, x):
        """
        标准前向传播，用于计算 Loss
        x: [B, Seq_Len]
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        # 训练时，Decoder 的输入是 x (通常整个序列输入，内部逻辑会自动处理，
        # 但严格来说应该输入 x[:, :-1]，预测 x[:, 1:]。
        # 为了方便，我们这里直接输入整个 x，计算 Loss 时注意错位即可)
        logits = self.decode(z, x)
        return logits, mu, logvar


# ============================================================================
# 模型二：基于 Transformer 的 VAE
# 优点：强大的特征提取能力，并行计算快
# 缺点：在短序列和小数据上容易过拟合，或忽略 z (Posterior Collapse)
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [B, Seq, Dim]
        return x + self.pe[:, :x.size(1)]

class MusicTransformerVAE(BaseVAE):
    def __init__(self, vocab_size=130, embed_dim=256, nhead=4, num_layers=2, latent_dim=128, seq_len=32):
        super(MusicTransformerVAE, self).__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=seq_len)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 降维到 Latent
        # 我们使用 Global Average Pooling 后的向量做映射
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_var = nn.Linear(embed_dim, latent_dim)
        
        # Decoder
        # 将 z 映射回 embedding 维度，作为 Memory 传入 Decoder
        self.z_to_memory = nn.Linear(latent_dim, embed_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def encode(self, x):
        # x: [B, Seq]
        src = self.embedding(x) * math.sqrt(self.embed_dim)
        src = self.pos_encoder(src)
        
        # Transformer 编码
        memory = self.transformer_encoder(src) # [B, Seq, Emb]
        
        # Global Average Pooling (把时间维取平均，得到整首曲子的概要)
        mean_memory = torch.mean(memory, dim=1) # [B, Emb]
        
        mu = self.fc_mu(mean_memory)
        logvar = self.fc_var(mean_memory)
        return mu, logvar

    def decode(self, z, x_input):
        # x_input: [B, Seq] (Target input)
        # z: [B, Latent]
        
        # 准备 Memory (来自 Latent)
        # 我们把 z 变成长度为 1 的序列，让 Decoder 去 Attention 它
        memory = self.z_to_memory(z).unsqueeze(1) # [B, 1, Emb]
        
        # 准备 Target Input
        tgt = self.embedding(x_input) * math.sqrt(self.embed_dim)
        tgt = self.pos_encoder(tgt)
        
        # 生成 Causal Mask (防止看到未来)
        seq_len = tgt.size(1)
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        tgt_mask = tgt_mask.to(z.device)
        
        # Decoder
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        logits = self.fc_out(output)
        return logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        # Transformer Decoder 需要 Shift 后的输入
        # 这里的 x 是完整的，我们在 Loss 计算时再处理 shift，
        # 但在 forward 里，decoder 看到的应该是 masked 的，所以输入完整 x 配合 causal mask 没问题
        logits = self.decode(z, x)
        return logits, mu, logvar