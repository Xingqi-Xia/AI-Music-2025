# ==============================================================================
# MusicGPT Training Script (Refined DDP Version)
# ==============================================================================

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch import amp
from tqdm import tqdm
import argparse
import math
import random


from model import MusicGPT, GPT_CONFIG 

from datasets import load_from_disk
from torch.nn import functional as F

# ==============================================================================
# ⚙️ Enable TF32 for faster training on compatible GPUs
# ==============================================================================
torch.backends.cuda.matmul.allow_tf32 = True # 允许矩阵乘法使用 TF32
torch.backends.cudnn.allow_tf32 = True       # 允许卷积使用 TF32
# ==============================================================================
# 🛠️ Configuration
# ==============================================================================
GPT_CONFIG_v3 = {
    "vocab_size": 130,
    "block_size": 512,
    "n_layer": 8,
    "n_head": 8,
    "embed_dim": 512,
    "dropout": 0.1,
    "bias": False,
}
GPT_CONFIG_nano = {
    "vocab_size": 130,
    "block_size": 64,
    "n_layer": 4,
    "n_head": 4,
    "embed_dim": 256,
    "dropout": 0.1,
    "bias": False,
}
GPT_CONFIG_standard = {
    "vocab_size": 130,
    "block_size": 128,
    "n_layer": 6,
    "n_head": 6,
    "embed_dim": 384,
    "dropout": 0.1,
    "bias": False,
}
GPT_CONFIG_heavy = {
    "vocab_size": 130,
    "block_size": 256,
    "n_layer": 8,
    "n_head": 8,
    "embed_dim": 512,
    "dropout": 0.1,
    "bias": False,
}

CONFIG_nano = {
    "exp_name": "music_gpt_nano_datav3",
    "data_path": "./dataset/gigamidi/gigamidi_processed_nodrums_v3", 
    "checkpoint_dir": "checkpoints_gpt",
    "log_dir": "logs_gpt_nano",
    "batch_size": 2048, # PER GPU batch size
    "epochs": 300,
    "learning_rate": 6e-4,
    "num_workers": 4, # A safe and efficient starting point for multi-GPU
    "model_config": GPT_CONFIG_nano,
    "warmup_iters": 2000,
    "lr_decay_iters": 600000,
    "min_lr": 4e-5,
    "eval_interval": 1,
}
CONFIG_standard = {
    "exp_name": "music_gpt_standard_datav3",
    "data_path": "./dataset/gigamidi/gigamidi_processed_nodrums_v3", 
    "checkpoint_dir": "checkpoints_gpt",
    "log_dir": "logs_gpt_standard",
    "batch_size": 1024, # PER GPU batch size
    "epochs": 400,
    "learning_rate": 8e-4,
    "num_workers": 4, # A safe and efficient starting point for multi-GPU
    "model_config": GPT_CONFIG_standard,
    "warmup_iters": 2000,
    "lr_decay_iters": 600000,
    "min_lr": 6e-5,
    "eval_interval": 1,
}
CONFIG_heavy = {
    "exp_name": "music_gpt_heavy_datav3",
    "data_path": "./dataset/gigamidi/gigamidi_processed_nodrums_v3", 
    "checkpoint_dir": "checkpoints_gpt",
    "log_dir": "logs_gpt_heavy",
    "batch_size": 1024, # PER GPU batch size
    "epochs": 400,
    "learning_rate": 8e-4,
    "num_workers": 4, # A safe and efficient starting point for multi-GPU
    "model_config": GPT_CONFIG_heavy,
    "warmup_iters": 2000,
    "lr_decay_iters": 600000,
    "min_lr": 6e-5,
    "eval_interval": 1,
}
CONFIG_v3 = {
    "exp_name": "music_gpt_gigamidi_v3_final",
    "data_path": "./dataset/gigamidi/gigamidi_processed_nodrums", 
    "checkpoint_dir": "checkpoints_gpt",
    "log_dir": "logs_gpt",
    "batch_size": 128, # PER GPU batch size
    "epochs": 200,
    "learning_rate": 6e-4,
    "num_workers": 4, # A safe and efficient starting point for multi-GPU
    "model_config": GPT_CONFIG_v3,
    "warmup_iters": 2000,
    "lr_decay_iters": 600000,
    "min_lr": 6e-5,
    "eval_interval": 1,
}
CONFIG=CONFIG_heavy

class SimpleMemoryDataset(Dataset):
    """
    一个简单的包装器，把 list of tensors 包装成类似 HF Dataset 的格式
    """
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 返回字典格式以兼容 DataCollator
        return {'input_ids': self.data[idx]}
# ==============================================================================
# 📦 Data Collator (Refined)
# ==============================================================================
class DataCollatorWithAugmentation:
    def __init__(self, block_size):
        self.block_size = block_size
        self.TOKEN_PAD = -100
        self.PITCH_OFFSET = 2
        self.VOCAB_SIZE = 130
        self.MIN_TRANSPOSE = -5
        self.MAX_TRANSPOSE = 6

    def __call__(self, batch):
        sequences = [item['input_ids'] for item in batch]
        processed_sequences = []
        
        # 【修正】我们需要多取一个 Token，用于生成 Target
        target_len = self.block_size + 1
        
        for seq in sequences:
            # 确保是 Tensor
            if not isinstance(seq, torch.Tensor):
                seq = torch.tensor(seq, dtype=torch.long)
                
            start_idx = 0
            seq_len = len(seq)
            
            # 随机切片
            if seq_len > target_len:
                start_idx = random.randint(0, seq_len - target_len)
                chunk = seq[start_idx : start_idx + target_len]
            else:
                chunk = seq # 如果太短，后面 padding 会处理

            # 移调增强
            transpose_val = random.randint(self.MIN_TRANSPOSE, self.MAX_TRANSPOSE)
            if transpose_val != 0:
                chunk = chunk.clone() # 避免修改原数据
                is_pitch = chunk >= self.PITCH_OFFSET
                chunk[is_pitch] += transpose_val
                chunk[is_pitch] = torch.clamp(chunk[is_pitch], self.PITCH_OFFSET, self.VOCAB_SIZE - 1)
            
            processed_sequences.append(chunk)

        padded_batch = torch.nn.utils.rnn.pad_sequence(
            processed_sequences, 
            batch_first=True, 
            padding_value=self.TOKEN_PAD
        )
        
        # 长度修正 (如果 batch 里全是短序列，或者为了满足 target_len)
        if padded_batch.shape[1] < target_len:
            pad_amount = target_len - padded_batch.shape[1]
            pad_tensor = torch.full((padded_batch.shape[0], pad_amount), self.TOKEN_PAD, dtype=torch.long)
            padded_batch = torch.cat([padded_batch, pad_tensor], dim=1)
        
        # 截断到准确长度
        padded_batch = padded_batch[:, :target_len]

        return padded_batch


# ==============================================================================
# 📈 Learning Rate Scheduler (Unchanged)
# ==============================================================================
def get_lr(it, config):
    if it < config["warmup_iters"]:
        return config["learning_rate"] * it / config["warmup_iters"]
    if it > config["lr_decay_iters"]:
        return config["min_lr"]
    decay_ratio = (it - config["warmup_iters"]) / (config["lr_decay_iters"] - config["warmup_iters"])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config["min_lr"] + coeff * (config["learning_rate"] - config["min_lr"])

# ==============================================================================
# 🚄 Training Main Function (Refactored for Robust DDP)
# ==============================================================================
def train(config, resume_path=None):
    # --- 1. DDP Setup (CRITICAL FIX) ---
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_ddp = world_size > 1
    ddp_rank = int(os.environ.get("RANK", "0"))
    ddp_local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if is_ddp:
        torch.cuda.set_device(ddp_local_rank)
        dist.init_process_group(
            backend="nccl",
            rank=ddp_rank,
            world_size=world_size,
        )
        device = f"cuda:{ddp_local_rank}"
        dist.barrier()
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Only the main process should create directories and print logs
    if ddp_rank == 0:
        print(f"Starting run: {config['exp_name']}")
        print(f"Using DDP: {is_ddp}, world_size: {world_size}, rank: {ddp_rank}, local_rank: {ddp_local_rank}")
        os.makedirs(config["checkpoint_dir"], exist_ok=True)
        os.makedirs(config["log_dir"], exist_ok=True)

    # torch.amp GradScaler to silence deprecation and be explicit about device
    # --- 2. Data Loading (支持 .pt 和 HF Disk) ---
    data_path = config["data_path"]
    scaler = amp.GradScaler(enabled=str(device).startswith("cuda"))
    if data_path.endswith('.pt'):
        if ddp_rank == 0: print(f"📦 Rank 0: Loading raw tensors from {data_path}...")
        
        # 加载 .pt 文件 (list of tensors)
        # 注意：DDP 中每个进程都会加载一份数据到内存。对于 10w 条 MIDI 来说内存占用不大，可以接受。
        raw_data = torch.load(data_path, map_location='cpu')
        
        # 手动打乱并划分
        # 为了保证多卡训练时大家划分的数据一致，我们需要固定种子
        random.seed(42) 
        random.shuffle(raw_data)
        
        split_idx = int(len(raw_data) * 0.9) # 90% 训练，10% 验证
        train_list = raw_data[:split_idx]
        val_list = raw_data[split_idx:]
        
        train_dataset = SimpleMemoryDataset(train_list)
        val_dataset = SimpleMemoryDataset(val_list)
        
        if ddp_rank == 0: print(f"   Loaded .pt file. Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
    else:
        # 走原有的 HF Dataset 逻辑
        if ddp_rank == 0: print("📦 Rank 0: Loading HF dataset from disk...")
        dataset_dict = load_from_disk(data_path)
        dataset_dict.set_format(type='torch', columns=['input_ids'])
        train_dataset = dataset_dict["train"]
        val_dataset = dataset_dict["validation"]
        if ddp_rank == 0: print(f"   Loaded HF disk. Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    data_collator = DataCollatorWithAugmentation(block_size=config["model_config"]["block_size"])
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_ddp else None

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], collate_fn=data_collator, 
        num_workers=config["num_workers"], pin_memory=pin_mem, shuffle=(train_sampler is None), sampler=train_sampler,
        persistent_workers=(config["num_workers"] > 0)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], collate_fn=data_collator, 
        num_workers=config["num_workers"], pin_memory=pin_mem, sampler=val_sampler,
        persistent_workers=(config["num_workers"] > 0)
    )
    
    if ddp_rank == 0:
        print(f"Data ready. Train samples: {len(dataset_dict['train'])}, Val samples: {len(dataset_dict['validation'])}")

    # --- 3. Model and Optimizer ---
    model = MusicGPT(config["model_config"]).to(device)
    if is_ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.95), fused=str(device).startswith("cuda")
    )

    # --- 4. Checkpoint Resumption ---
    start_epoch, best_val_loss, iter_num = 0, float('inf'), 0
    raw_model = model.module if is_ddp else model
    
    if resume_path and os.path.exists(resume_path):
        if ddp_rank == 0: print(f"🔄 Resuming training from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        raw_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        iter_num = checkpoint.get('iter_num', 0)
        if ddp_rank == 0: print(f"   Resumed from epoch {start_epoch}, iter {iter_num}, best val loss {best_val_loss:.4f}")

    # --- 5. Logging and Training Loop ---
    writer = SummaryWriter(log_dir=os.path.join(config["log_dir"], config["exp_name"])) if ddp_rank == 0 else None

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        if is_ddp: train_sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}", disable=(ddp_rank != 0))
        for batch in pbar:
            lr = get_lr(iter_num, config)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            X = batch[:, :-1].to(device, non_blocking=True)
            Y = batch[:, 1:].to(device, non_blocking=True)

            # Safety: clamp inputs to valid vocab, set padding to ignore_index
            X = X.clone()
            Y = Y.clone()
            X[X < 0] = 0
            X.clamp_(0, config["model_config"]["vocab_size"] - 1)
            Y[Y < 0] = -1
            Y[Y >= config["model_config"]["vocab_size"]] = -1

            with amp.autocast(device_type="cuda", enabled=str(device).startswith("cuda")):
                logits, loss = model(X, targets=Y)
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if ddp_rank == 0:
                pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{lr:.6f}"})
                writer.add_scalar("Train/Loss_step", loss.item(), iter_num)
                writer.add_scalar("Train/LearningRate", lr, iter_num)
            iter_num += 1

        # --- 6. Validation and Checkpointing ---
        if (epoch + 1) % config["eval_interval"] == 0:
            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc="Validating", disable=(ddp_rank != 0))
                for batch in val_pbar:
                    X = batch[:, :-1].to(device, non_blocking=True)
                    Y = batch[:, 1:].to(device, non_blocking=True)
                    X = X.clone()
                    Y = Y.clone()
                    X[X < 0] = 0
                    X.clamp_(0, config["model_config"]["vocab_size"] - 1)
                    Y[Y < 0] = -1
                    Y[Y >= config["model_config"]["vocab_size"]] = -1
                    _, loss = model(X, targets=Y)
                    val_loss_sum += loss.item()
            
            if is_ddp:
                val_stats = torch.tensor([val_loss_sum, len(val_loader)], device=device)
                dist.all_reduce(val_stats, op=dist.ReduceOp.SUM)
                avg_val_loss = (val_stats[0] / val_stats[1]).item()
            else:
                avg_val_loss = val_loss_sum / len(val_loader)

            if ddp_rank == 0:
                writer.add_scalar("Val/Loss_epoch", avg_val_loss, epoch)
                print(f"\n📊 Validation Loss (Epoch {epoch+1}): {avg_val_loss:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    checkpoint = {
                        'epoch': epoch, 'model_state_dict': raw_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_loss': best_val_loss, 'iter_num': iter_num, 'config': config,
                    }
                    best_path = os.path.join(config["checkpoint_dir"], f"{config['exp_name']}_best.pth")
                    torch.save(checkpoint, best_path)
                    print(f"🌟 New best model saved to {best_path}")
            latest_path = os.path.join(config["checkpoint_dir"], f"{config['exp_name']}_latest.pth")
            checkpoint = {
                'epoch': epoch, 'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss, 'iter_num': iter_num, 'config': config,
            }
            torch.save(checkpoint, latest_path)
    # --- 7. Cleanup ---
    if writer is not None: writer.close()
    if is_ddp: dist.destroy_process_group()
    if ddp_rank == 0: print("🏁 Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a MusicGPT model.")
    parser.add_argument('--resume', type=str, default=None, help="Path to a checkpoint. Use 'latest' for the latest.")
    args = parser.parse_args()

    resume_path = None
    if args.resume:
        if args.resume.lower() == 'latest':
            resume_path = os.path.join(CONFIG["checkpoint_dir"], f"{CONFIG['exp_name']}_latest.pth")
            if not os.path.exists(resume_path):
                print(f"Warning: 'latest' checkpoint not found at {resume_path}. Starting from scratch.")
                resume_path = None
        else:
            resume_path = args.resume
    
    train(CONFIG, resume_path)
# python -m torch.distributed.run --standalone --nproc_per_node=3 trainv2.py