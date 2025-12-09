import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# A recommended configuration for a "small" but powerful model
# that will train fast on an A100.
GPT_CONFIG = {
    "vocab_size": 132,      # 0-129 for music tokens, + BOS, EOS, PAD etc. if you add them
    "block_size": 256,      # Context window size
    "n_layer": 8,           # Number of Transformer blocks
    "n_head": 8,            # Number of attention heads
    "embed_dim": 512,       # Embedding dimension
    "dropout": 0.1,         # Dropout rate
}


class MusicGPT(nn.Module):
    """
    A GPT-style Transformer model for music generation and evaluation.
    This model is autoregressive, meaning it predicts the next token in a sequence.
    """
    def __init__(self, config=GPT_CONFIG):
        super().__init__()
        assert config["vocab_size"] is not None
        assert config["block_size"] is not None
        self.config = config

        # Token and Position embeddings
        self.token_embedding = nn.Embedding(config["vocab_size"], config["embed_dim"])
        self.position_embedding = nn.Embedding(config["block_size"], config["embed_dim"])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config["dropout"])
        
        # The core of the model: a stack of Transformer blocks
        # We use PyTorch's efficient implementation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["embed_dim"],
            nhead=config["n_head"],
            dim_feedforward=4 * config["embed_dim"], # Standard practice
            dropout=config["dropout"],
            batch_first=True, # Critical for easy tensor manipulation
            activation=F.gelu
        )
        self.transformer_blocks = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config["n_layer"]
        )

        # Final Layer Normalization before the output head
        self.ln_f = nn.LayerNorm(config["embed_dim"])
        
        # The final linear layer that maps hidden states to vocabulary logits
        self.lm_head = nn.Linear(config["embed_dim"], config["vocab_size"], bias=False)

        # Weight tying: share weights between embedding and final layer
        # This is a common practice that improves performance and reduces parameters.
        self.token_embedding.weight = self.lm_head.weight

        # Causal mask to ensure attention is only applied to the left in the input sequence
        # We register it as a buffer so it's not considered a model parameter.
        self.register_buffer("causal_mask", torch.triu(torch.ones(config["block_size"], config["block_size"]) * float('-inf'), diagonal=1))


    def forward(self, idx, targets=None):
        """
        Defines the forward pass of the model.
        
        Args:
            idx (torch.Tensor): Input sequence of token IDs. Shape: (B, T)
            targets (torch.Tensor, optional): Target sequence for loss calculation. Shape: (B, T)
        
        Returns:
            logits (torch.Tensor): Output logits. Shape: (B, T, vocab_size)
            loss (torch.Tensor, optional): The cross-entropy loss if targets are provided.
        """
        B, T = idx.size()
        assert T <= self.config["block_size"], f"Cannot forward sequence of length {T}, max is {self.config['block_size']}"

        # 1. Get token and position embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, embed_dim)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.position_embedding(pos) # (T, embed_dim) -> broadcasting adds B dimension
        
        # 2. Combine embeddings and apply dropout
        x = self.dropout(tok_emb + pos_emb)
        
        # 3. Pass through Transformer blocks with the causal mask
        # We slice the mask to match the input sequence length T
        x = self.transformer_blocks(x, mask=self.causal_mask[:T, :T])
        
        # 4. Final layer normalization and linear head
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # 5. Calculate loss if targets are provided
        loss = None
        if targets is not None:
            # Reshape for CrossEntropyLoss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively.
        
        Args:
            idx (torch.Tensor): The starting sequence (prompt). Shape: (B, T)
            max_new_tokens (int): The number of new tokens to generate.
            temperature (float): Softmax temperature for sampling. Higher -> more random.
            top_k (int, optional): Sample from the top_k most likely tokens.
            
        Returns:
            torch.Tensor: The generated sequence. Shape: (B, T + max_new_tokens)
        """
        self.eval() # Set model to evaluation mode
        for _ in range(max_new_tokens):
            # Crop context if it exceeds block size
            idx_cond = idx if idx.size(1) <= self.config["block_size"] else idx[:, -self.config["block_size"]:]
            
            # Forward pass to get logits for the next token
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature # Get the last logit and apply temperature
            
            # Optional Top-K sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Get probabilities and sample the next token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append the new token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        self.train() # Set model back to training mode
        return idx


if __name__ == '__main__':
    # This block is for testing the model definition
    print("Testing MusicGPT model definition...")
    
    # Use the default config
    config = GPT_CONFIG
    model = MusicGPT(config)
    
    # Create a dummy input batch
    # Batch size = 4, Sequence length = 128
    dummy_input = torch.randint(0, config["vocab_size"], (4, 128))
    
    # Forward pass
    logits, loss = model(dummy_input, targets=dummy_input)
    
    print(f"Model instantiated successfully.")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Calculated loss: {loss.item() if loss is not None else 'N/A'}")
    
    # Test generation
    print("\nTesting generation...")
    start_tokens = torch.zeros((1, 1), dtype=torch.long) # Start with a single 'rest' token
    generated_sequence = model.generate(start_tokens, max_new_tokens=31)
    
    print(f"Generated sequence shape: {generated_sequence.shape}")
    print(f"Generated sequence (first 32 tokens): {generated_sequence.squeeze().tolist()}")
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal number of parameters: {num_params/1e6:.2f}M")