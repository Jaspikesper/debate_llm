import torch
from model import *
from Debugging.Encoding.model import MaskedSelfAttention
from tokenizer import BytePairTokenizer, TikToken4o
import math
from tokenizer import BytePairTokenizer

# Data hyperparameters
h_dim = 18
max_length = 6

# Initialize tokenizer and embedding layer
tokenizer = BytePairTokenizer(vocab_file='byte_training/tok.json')
vocab_size = max(tokenizer.vocab.values()) + 2
embedding_layer = torch.nn.Embedding(vocab_size, h_dim, padding_idx=0)

# Positional Encoding
class PositionalEncoding(torch.nn.Module):
    def __init__(self, max_length, h_dim):
        super().__init__()
        self.pos_embedding = torch.nn.Embedding(max_length, h_dim)

    def forward(self, x):
        # x: (batch_size, seq_length)
        seq_length = x.size(1)
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0).expand_as(x)
        return self.pos_embedding(positions)

# Encoding Module (Embeddings + PositionalEncoding)
class Encoding(torch.nn.Module):
    def __init__(self, embedding_layer, pos_encoding):
        super().__init__()
        self.embedding = embedding_layer
        self.pos_encoding = pos_encoding

    def forward(self, x):
        embeds = self.embedding(x)
        pos_embeds = self.pos_encoding(x)
        return embeds + pos_embeds

# Masked Self Attention
class MaskedSelfAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout=0.5, qkv_bias=True):
        super().__init__()
        self.W_q = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.scale = 1 / math.sqrt(d_out)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length)) * -1e9)

    def forward(self, x):
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)
        attn_scores = (queries @ keys.transpose(-2, -1)) * self.scale
        seq_len = x.size(1)
        mask = self.mask[:seq_len, :seq_len]
        attn_scores = attn_scores + mask
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        z = attn_weights @ values
        z = self.dropout(z)
        return z

# Example input
phrase = 'to pimp a '
tokens = tokenizer.encode(phrase)
input_tensor = torch.tensor(tokens).unsqueeze(0)  # (1, seq_length)

# Initialize abstract layers
pos_encoding_layer = PositionalEncoding(max_length=max_length, h_dim=h_dim)
encoding_layer = Encoding(embedding_layer, pos_encoding_layer)
d_out = 16
attention_layer = MaskedSelfAttention(d_in=h_dim, d_out=d_out, context_length=max_length)

# Call them in order
embedded_with_pos = encoding_layer(input_tensor)
context = attention_layer(embedded_with_pos)

print("Output shape (context):", context.shape)

class MyStackedFunction(torch.nn.Module):
    def __init__(self, module_cls, N, *args, **kwargs):
        super().__init__()
        self.modules_list = torch.nn.ModuleList([module_cls(*args, **kwargs) for _ in range(N)])

    def forward(self, x):
        outputs = [mod(x) for mod in self.modules_list]
        return torch.stack(outputs, dim=0).squeeze()

num_heads = 12
StackedAttention = MyStackedFunction(
    MaskedSelfAttention,
    num_heads,
    h_dim,
    d_out,
    max_length,
    dropout=0.5,
    qkv_bias=True
)
