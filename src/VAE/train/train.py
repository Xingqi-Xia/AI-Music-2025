# 从数据集随便读取一列，然后把它import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
import random
import os
# 导入你的模型,目前这里用的是GRU架构
from models import MusicGRUVAE
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ==============================================================================
# 超参数配置 
# ==============================================================================
CONFIG = {
    "exp_name": "vae_gru_bach_v2",     # 实验名称
    "data_path": "classical_dataset.pt",
    
    # 训练参数
    "batch_size": 3072,                
    "epochs": 100,                     # 训练轮数
    "learning_rate": 1e-3,
    "num_workers": 16,                 # 数据加载线程数
    
    # 模型参数 (必须与 model.py 保持一致)
    "vocab_size": 130,                 # 0-129
    "embed_dim": 256,
    "hidden_dim": 512,
    "latent_dim": 128,
    "seq_len": 32,
    
    # KL 退火策略
    "kl_start_epoch": 3,               # 前 5 个 epoch 不算 KL Loss (让模型先学会重构)
    "kl_anneal_cycle": 40,             # 用 50 个 epoch 把 beta 从 0 增加到 1
    "beta_max": 0.02,                   # KL 权重的上限 (太高会导致重构变差，对于我们的离散序列任务，最好不要高于0.1)
}

# ==============================================================================
# 🚀 辅助函数
# ==============================================================================

def loss_function(logits, target, mu, logvar, beta):
    """
    计算 VAE 的总 Loss
    logits: [B, Seq_Len-1, Vocab]
    target: [B, Seq_Len-1]
    mu, logvar: [B, Latent]
    beta: KL 散度的权重
    """
    # 1. 重构损失 (Reconstruction Loss)
    # Flatten 到 [B * Seq, Vocab] 以计算 CrossEntropy
    recon_loss = nn.CrossEntropyLoss()(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
    
    # 2. KL 散度 (KL Divergence)
    # 公式: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # 平均到每个样本 (batch mean)
    kl_loss = kl_loss / logits.size(0)
    
    # 总损失
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss

def get_beta(epoch, config):
    """
    计算当前 Epoch 的 KL 权重 (Cyclic Annealing 或 Linear Annealing)
    这里使用简单的 Linear Annealing
    """
    if epoch < config["kl_start_epoch"]:
        return 0.0
    
    # 线性增长
    steps = epoch - config["kl_start_epoch"]
    beta = min(config["beta_max"], (steps / config["kl_anneal_cycle"]) * config["beta_max"])
    return beta

# ==============================================================================
# 🚄 训练主流程
# ==============================================================================

def main():
    # 1. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 使用设备: {device} (GPU数量: {torch.cuda.device_count()})")
    
    # 2. 加载数据
    print(f"📦 加载数据集: {CONFIG['data_path']} ...")
    full_data = torch.load(CONFIG['data_path'])
    
    # 简单的 Train/Val 切分 (95% 训练, 5% 验证)
    train_size = int(0.95 * len(full_data))
    val_size = len(full_data) - train_size
    train_dataset, val_dataset = random_split(full_data, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=False, 
        num_workers=CONFIG["num_workers"],
        pin_memory=True
    )
    
    print(f"✅ 数据加载完成. 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

    # 3. 初始化模型
    model = MusicGRUVAE(
        vocab_size=CONFIG["vocab_size"],
        embed_dim=CONFIG["embed_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        latent_dim=CONFIG["latent_dim"],
        seq_len=CONFIG["seq_len"]
    ).to(device)

    # 多卡并行
    if torch.cuda.device_count() > 1:
        print("⚡ 启用 DataParallel 多卡训练")
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    scaler = GradScaler(enabled=device.type == "cuda") # 混合精度训练（CPU 时自动关闭）
    writer = SummaryWriter(log_dir=f"logs/{CONFIG['exp_name']}")

    # 4. 训练循环
    best_val_loss = float("inf")
    
    for epoch in range(CONFIG["epochs"]):
        model.train()
        running_recon = 0.0
        running_kl = 0.0
        
        # 获取当前的 beta
        beta = get_beta(epoch, CONFIG)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [β={beta:.4f}]")
        
        for batch in pbar:
            batch = batch.to(device) # [B, 32]
            #print("正在训练批次，批次大小：", batch.size(0))
            
            # --- 关键：Next Token Prediction ---
            # Encoder 输入: 完整的序列 (0 ~ 31)
            # Decoder 输入: 完整的序列 (模型内部处理 Teacher Forcing)
            # Target (标签): 应该是输入向左移动一位
            #   Input: [A, B, C, D]
            #   Logits 对应: [pred_B, pred_C, pred_D, pred_E]
            #   所以我们需要拿 logits[:, :-1] 和 batch[:, 1:] 比较
            #   即：看到 A 预测 B，看到 B 预测 C...
            
            optimizer.zero_grad()
            
            # 使用 torch.amp.autocast 兼容新版本；CPU 时自动关闭混合精度
            with autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cuda" else None):
                logits, mu, logvar = model(batch) 
                
                # 对齐 Logits 和 Targets
                # logits: [B, 32, Vocab] -> 取前 31 个预测
                logits_pred = logits[:, :-1, :]
                # targets: [B, 32] -> 取后 31 个真实值
                targets = batch[:, 1:]
                
                loss, recon, kl = loss_function(logits_pred, targets, mu, logvar, beta)
            #print("计算损失完成，开始反向传播")
            # 反向传播
            scaler.scale(loss).backward()
            
            # 梯度裁剪 (防止梯度爆炸)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            #print("优化器步骤完成")
            # 记录
            running_recon += recon.item()
            running_kl += kl.item()
            
            pbar.set_postfix({"Recon": f"{recon.item():.4f}", "KL": f"{kl.item():.4f}"})

        # --- Epoch 结束记录 ---
        avg_recon = running_recon / len(train_loader)
        avg_kl = running_kl / len(train_loader)
        
        writer.add_scalar("Train/Recon_Loss", avg_recon, epoch)
        writer.add_scalar("Train/KL_Loss", avg_kl, epoch)
        writer.add_scalar("Train/Beta", beta, epoch)

        # --- 验证循环 ---
        model.eval()
        val_recon = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits, mu, logvar = model(batch)
                
                logits_pred = logits[:, :-1, :]
                targets = batch[:, 1:]
                
                _, recon, _ = loss_function(logits_pred, targets, mu, logvar, beta=1.0) # 验证时通常看纯粹的指标
                val_recon += recon.item()
        
        avg_val_loss = val_recon / len(val_loader)
        writer.add_scalar("Val/Recon_Loss", avg_val_loss, epoch)
        
        print(f"📊 Validation Loss: {avg_val_loss:.4f}")

        # --- 保存模型 ---
        # 保存最新
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, f"checkpoints/{CONFIG['exp_name']}_latest.pth")
        
        # 保存最佳
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, f"checkpoints/{CONFIG['exp_name']}_best.pth")
            print("🌟 New Best Model Saved!")

    writer.close()
    print("🏁 训练完成！")

if __name__ == "__main__":
    # 创建文件夹
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    main()