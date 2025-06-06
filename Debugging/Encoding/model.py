import torch
from tokenizer import BytePairTokenizer
from data_loader import GPT_Dataset, make_dataloader
import math

with open('byte_training/LLM_dataset.txt', 'r', encoding='utf-8') as f:
    data = f.read()
t = BytePairTokenizer(vocab_file='byte_training/tok.json')

# Hyperparameters for embedding matrix
vocab_size = max(t.vocab.values()) + 2
h_dim = 3

# Parameters for dataloader
shuffle = False
drop_last = False
num_workers = 0
batch_size = 4
stride = 1
max_length = 6

# Initialize embedder, dataset, and most importantly data loader
embedding_layer = torch.nn.Embedding(vocab_size, h_dim, padding_idx=0)
dataset = GPT_Dataset(data, t, max_length=max_length, stride=stride)
data_loader = make_dataloader(dataset, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, batch_size=batch_size)

inputs, targets = next(iter(data_loader))
max_token_id = inputs.max().item()
if max_token_id >= vocab_size:
    print(f"ERROR: Max token ID ({max_token_id}) >= vocab_size ({vocab_size})")
    print("This will cause an IndexError!")

def display_task():
    print("Displaying the sequence task!")
    print("Decoded Inputs:")
    for seq in inputs:
        print(t.decode(seq.tolist()))

    print("\nDecoded Targets:")
    for seq in targets:
        print(t.decode(seq.tolist()))

display_task()

class PositionalEncoding(torch.nn.Module):
    def __init__(self, max_length, h_dim):
        super().__init__()
        self.pos_embedding = torch.nn.Embedding(max_length, h_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_length)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        pos_emb = self.pos_embedding(positions)
        return pos_emb

class Encoding(torch.nn.Module):  # The model's positional embedding module
    def __init__(self, embedding_layer, positional_encoding):
        super().__init__()
        self.embedding = embedding_layer
        self.pos_encoding = positional_encoding

    def forward(self, x):
        x = self.embedding(x)  # Shape: (batch_size, seq_length, h_dim)
        x = x + self.pos_encoding(x.argmax(dim=-1))  # Passing x's token indices to pos_encoding to get pos embeddings
        return x

class MaskedSelfAttention(torch.nn.Module):
    def __init__(self, h_dim, qkv_bias=True):
        super().__init__()
        self.W_q = torch.nn.Linear(h_dim, h_dim, bias=qkv_bias)
        self.W_k = torch.nn.Linear(h_dim, h_dim, bias=qkv_bias)
        self.W_v = torch.nn.Linear(h_dim, h_dim, bias=qkv_bias)
        self.scale = 1/math.sqrt(h_dim)

    def mask(self, A):
        one = torch.ones_like(A)
        triu = 1-torch.tril(one) # Diagonal is zero
        mask = triu * -9999
        masked = mask + A
        return masked

    def forward(self, x):
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)
        z = torch.nn.functional.softmax(
            self.mask(queries @ keys.transpose(-2, -1)) * self.scale,
            dim=1
        ) @ values

        return z



pos_encoding_layer = PositionalEncoding(max_length, h_dim)
encoding_module = Encoding(embedding_layer, pos_encoding_layer)

# Example test forward pass
encoded = encoding_module(inputs)  # shape should be (batch_size, seq_length, h_dim)
print(f'Encoded output shape: {encoded.shape}')
self_attention_layer = MaskedSelfAttention(h_dim=h_dim)
context = self_attention_layer(encoded)
encoded += context # Encoded pays attention to its context.



