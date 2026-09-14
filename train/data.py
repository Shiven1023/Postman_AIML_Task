import torch

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))

    stoi = {ch: i for i, ch in enumerate(chars)} 
    itos = {i: ch for i, ch in enumerate(chars)}

    data = torch.tensor([stoi[ch] for ch in text],dtype=torch.long)
    return data, stoi, itos

def train_val_split(data, split=0.9):
    n = int(split * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data


def get_batch(data, batch_size, context_length, device):
    ix = torch.randint(0,len(data) - context_length - 1,(batch_size,))
    x = torch.stack([data[i:i + context_length]for i in ix])
    y = torch.stack([data[i + 1:i + context_length + 1]for i in ix])

    return x.to(device), y.to(device)

if __name__ == "__main__":
    data, stoi, itos = load_text("data/tiny-shakespeare.txt")

    train_data, val_data = train_val_split(data)
    x, y = get_batch(train_data,batch_size=4,context_length=16,device="cpu")

    print("Vocabulary size:", len(stoi))
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("Input:")
    print("".join(itos[int(i)] for i in x[0]))
    print("Target:")
    print("".join(itos[int(i)] for i in y[0]))