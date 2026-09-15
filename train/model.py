import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from attention.dense import MultiHeadAttention
from attention.sliding_window import SlidingWindowAttention
from attention.BigBird import BigBirdSparse

from train.data import load_text, train_val_split, get_batch

class CausalDenseAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d, n_heads)
    def forward(self, x):
        N = x.shape[1]
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=x.device),diagonal=1)

        return self.attn(x, mask)


class DecoderBlock(nn.Module):
    def __init__(self, d, attention):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attention = attention
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4*d),
            nn.GELU(),
            nn.Linear(4*d, d)
        )

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x



class CharacterGPT(nn.Module):

    def __init__(self, vocab_size , context_length ,d , attention1, attention2):
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size , d)

        self.position_embedding = nn.Embedding(context_length , d)
        self.block1 = DecoderBlock(d, attention1)
        self.block2 = DecoderBlock(d, attention2)
        self.final_ln = nn.LayerNorm(d)
        self.final_linear = nn.Linear(d, vocab_size)

    def forward(self,input_seq, targets = None):
        B, N = input_seq.shape
        #(B,N) --> (B,N,d)
        token_emb = self.token_embedding(input_seq)
        positions = torch.arange(N,device=input_seq.device)
        # (N) -> (N,d)
        position_emb = self.position_embedding(positions)

        x = token_emb + position_emb
        x = self.block1(x)
        x = self.block2(x)

        x = self.final_ln(x)

        # (B,N,d) -> (B,N,vocab_size)
        logits = self.final_linear(x)
        loss = None

        if targets is not None:

            # (B,N,vocab_size) -> (B*N,vocab_size)
            logits_flat = logits.reshape(B * N,self.vocab_size)
            # (B,N) -> (B*N)
            targets_flat = targets.reshape(B * N)

            loss = F.cross_entropy(logits_flat,targets_flat)
        return logits, loss
        

@torch.no_grad()
def evaluate(
    model,
    val_data,
    batch_size,
    context_length,
    device,
    num_batches=20
):
    model.eval()

    losses = []

    for _ in range(num_batches):

        x, y = get_batch(
            val_data,
            batch_size,
            context_length,
            device
        )

        _, loss = model(x, y)

        losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)


# Train one model
def train_model(
    name,
    model,
    train_data,
    val_data,
    batch_size,
    context_length,
    device,
    steps,
    learning_rate
):

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    model.train()

    print(f"\nTraining {name}")

    for step in range(steps):

        x, y = get_batch(
            train_data,
            batch_size,
            context_length,
            device
        )

        _, loss = model(x, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if step % 100 == 0:
            print(
                f"Step {step:4d} | "
                f"Train loss: {loss.item():.4f}"
            )

    val_loss = evaluate(
        model,
        val_data,
        batch_size,
        context_length,
        device
    )

    print(
        f"{name} validation loss: "
        f"{val_loss:.4f}"
    )

    return val_loss


# Main


if __name__ == "__main__":
    device = ("cuda" if torch.cuda.is_available()else "cpu")

    print("Device:", device)

    # Hyperparameters
    context_length = 256
    batch_size = 32
    d = 128
    n_heads = 4
    learning_rate = 3e-4
    steps = 5
    window_size = 64
    block_size = 32
    num_local_blocks = 1
    num_random_blocks = 1

    # Data
    data, stoi, itos = load_text("data/tiny-shakespeare.txt")
    train_data, val_data = train_val_split(data)
    vocab_size = len(stoi)
    print("Vocabulary size:", vocab_size)
    results = {}
    # Dense

    torch.manual_seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    dense_model = CharacterGPT(
        vocab_size,
        context_length,
        d,

        CausalDenseAttention(
            d,
            n_heads
        ),

        CausalDenseAttention(
            d,
            n_heads
        )
    )

    results["Dense"] = train_model(
        "Dense",
        dense_model,
        train_data,
        val_data,
        batch_size,
        context_length,
        device,
        steps,
        learning_rate
    )

    del dense_model

    if device == "cuda":
        torch.cuda.empty_cache()

    # Sliding window

    torch.manual_seed(42)
    random.seed(42)
    sliding_model = CharacterGPT(
        vocab_size,
        context_length,
        d,

        SlidingWindowAttention(
            d,
            n_heads,
            window_size
        ),

        SlidingWindowAttention(
            d,
            n_heads,
            window_size
        )
    )

    results["Sliding"] = train_model(
        "Sliding",
        sliding_model,
        train_data,
        val_data,
        batch_size,
        context_length,
        device,
        steps,
        learning_rate
    )

    del sliding_model

    if device == "cuda":
        torch.cuda.empty_cache()

    # BigBird-style

    torch.manual_seed(42)
    random.seed(42)

    bigbird_model = CharacterGPT(
        vocab_size,
        context_length,
        d,

        BigBirdSparse(
            d,
            n_heads,
            block_size,
            num_local_blocks,
            num_random_blocks
        ),

        BigBirdSparse(
            d,
            n_heads,
            block_size,
            num_local_blocks,
            num_random_blocks
        )
    )

    results["BigBird"] = train_model(
        "BigBird",
        bigbird_model,
        train_data,
        val_data,
        batch_size,
        context_length,
        device,
        steps,
        learning_rate
    )
    print("\n FINAL LOSS(VALIDATION)")

    for name, loss in results.items():
        print(f"{name:10s}: {loss:.4f}")
