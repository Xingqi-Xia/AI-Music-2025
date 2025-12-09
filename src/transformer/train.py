import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import argparse
import math

# Import your model definition
from model import MusicGPT, GPT_CONFIG

# ==============================================================================
# 🛠️ Configuration
# ==============================================================================
# These can be overridden by command-line arguments if needed
CONFIG = {
    "exp_name": "music_gpt_v1",
    "data_path": "./dataset/classical_gpt_dataset_smart_v2.pt",
    "checkpoint_dir": "checkpoints_gpt",
    "log_dir": "logs_gpt",
    
    # Training parameters
    "batch_size": 1024,  # Per GPU. A100 can handle this easily.
    "epochs": 300,
    "learning_rate": 3e-4, # A good starting point for AdamW
    "num_workers": 16,
    
    # Model parameters (should match model.py)
    "model_config": GPT_CONFIG,
    
    # LR Scheduler parameters
    "warmup_iters": 200,    # How many steps to warm up for
    "lr_decay_iters": 50000, # Should be ~= total number of training steps
    "min_lr": 3e-5,       # Lower bound for LR
    
    # Logging/Saving
    "eval_interval": 1,     # Evaluate on validation set every N epochs
}

# ==============================================================================
# 📦 Custom Dataset for Dynamic Slicing
# ==============================================================================

class MusicGPTDataset(Dataset):
    """
    Takes a list of long sequences and serves up random chunks of `block_size`.
    """
    def __init__(self, long_sequences, block_size):
        self.long_sequences = long_sequences
        self.block_size = block_size

    def __len__(self):
        # A rough estimate of total possible samples.
        # In reality, we sample randomly, so this is just for DataLoader.
        return sum(len(seq) for seq in self.long_sequences) // self.block_size

    def __getitem__(self, idx):
        # Instead of using idx, we just pick a random sequence and a random chunk.
        # This is a common and effective strategy for training on long text/music.
        long_seq = self.long_sequences[torch.randint(len(self.long_sequences), (1,))]
        
        start_idx = 0
        if len(long_seq) > self.block_size:
            start_idx = torch.randint(0, len(long_seq) - self.block_size, (1,))
            
        chunk = long_seq[start_idx : start_idx + self.block_size]
        
        # Pad if the sequence is shorter than block_size
        # Using a value like 0 (Rest) for padding is reasonable here
        if len(chunk) < self.block_size:
            padding = torch.zeros(self.block_size - len(chunk), dtype=torch.long)
            chunk = torch.cat([torch.tensor(chunk, dtype=torch.long), padding], dim=0)
        else:
            chunk = torch.tensor(chunk, dtype=torch.long)

        return chunk

# ==============================================================================
# 📈 Learning Rate Scheduler
# ==============================================================================

def get_lr(it, config):
    # 1) linear warmup for warmup_iters steps
    if it < config["warmup_iters"]:
        return config["learning_rate"] * it / config["warmup_iters"]
    # 2) if it > lr_decay_iters, return min learning rate
    if it > config["lr_decay_iters"]:
        return config["min_lr"]
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - config["warmup_iters"]) / (config["lr_decay_iters"] - config["warmup_iters"])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return config["min_lr"] + coeff * (config["learning_rate"] - config["min_lr"])

# ==============================================================================
# 🚄 Training Main Function
# ==============================================================================

def train(config, resume_path=None):
    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)
    
    scaler = GradScaler()

    # --- 2. Data Loading ---
    print("📦 Loading and preparing dataset...")
    all_sequences = torch.load(config["data_path"],weights_only=False)
    
    # Split data
    train_size = int(0.95 * len(all_sequences))
    val_size = len(all_sequences) - train_size
    train_data, val_data = random_split(all_sequences, [train_size, val_size])

    train_dataset = MusicGPTDataset(list(train_data), config["model_config"]["block_size"])
    val_dataset = MusicGPTDataset(list(val_data), config["model_config"]["block_size"])
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"], pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"], pin_memory=True)
    print(f"✅ Data ready. Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # --- 3. Model and Optimizer ---
    model = MusicGPT(config["model_config"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.95))

    # --- 4. Checkpoint Resumption ---
    start_epoch = 0
    best_val_loss = float('inf')
    iter_num = 0

    if resume_path:
        print(f"🔄 Resuming training from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        iter_num = checkpoint.get('iter_num', 0)
        print(f"   Resumed from epoch {start_epoch}, best validation loss was {best_val_loss:.4f}")

    # --- 4.1 TensorBoard Writer (placed after resume to keep step continuity) ---
    writer = SummaryWriter(
        log_dir=os.path.join(config["log_dir"], config["exp_name"]),
        purge_step=iter_num  # 保持 step 连续，TensorBoard 会从该 step 继续
    )

    # --- 5. Training Loop ---
    for epoch in range(start_epoch, config["epochs"]):
        # --- Training Phase ---
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch in pbar:
            lr = get_lr(iter_num, config)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            X = batch[:, :-1].to(device)
            Y = batch[:, 1:].to(device)

            with autocast():
                logits, loss = model(X, targets=Y)
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{lr:.6f}"})
            writer.add_scalar("Train/Loss", loss.item(), iter_num)
            writer.add_scalar("Train/LearningRate", lr, iter_num)
            iter_num += 1

        # --- Validation Phase ---
        if (epoch + 1) % config["eval_interval"] == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    X = batch[:, :-1].to(device)
                    Y = batch[:, 1:].to(device)
                    _, loss = model(X, targets=Y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            writer.add_scalar("Val/Loss", avg_val_loss, epoch)
            print(f"\n📊 Validation Loss: {avg_val_loss:.4f}")

            # --- Save Checkpoint ---
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'iter_num': iter_num,
                'config': config,
            }
            # Save latest checkpoint
            latest_path = os.path.join(config["checkpoint_dir"], f"{config['exp_name']}_latest.pth")
            torch.save(checkpoint, latest_path)

            # Save best checkpoint
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint['best_val_loss'] = best_val_loss
                best_path = os.path.join(config["checkpoint_dir"], f"{config['exp_name']}_best.pth")
                torch.save(checkpoint, best_path)
                print(f"🌟 New best model saved to {best_path}")

    writer.close()
    print("🏁 Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a MusicGPT model.")
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to a checkpoint to resume from. If 'latest', resumes from the latest checkpoint.")
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