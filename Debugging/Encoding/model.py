import os
import sys
import torch
import torch.nn as nn
import math

from Debugging.Encoding.tokenizer import TikToken200k
from tokenizer import BytePairTokenizer
from model_config import GPT_CONFIG_124M


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class PositionalEncoding(nn.Module):
    def __init__(self, max_length: int, h_dim: int):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_length, h_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_length = x.size(1)
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0).expand_as(x)
        return self.pos_embedding(positions)

class Encoding(nn.Module):
    def __init__(self, embedding_layer: nn.Embedding, pos_encoding: PositionalEncoding):
        super().__init__()
        self.embedding = embedding_layer
        self.pos_encoding = pos_encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        embeds = self.embedding(x)
        pos_embeds = self.pos_encoding(x)
        return embeds + pos_embeds


class MaskedSelfAttention(nn.Module):
    def __init__(self, d_in: int, attn_dim: int, context_length: int, dropout: float = 0.1, qkv_bias: bool = False):
        super().__init__()
        self.proj = nn.Linear(d_in, 3 * attn_dim, bias=qkv_bias)
        self.attn_dim = attn_dim
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.proj(x)
        q, k, v = qkv.split(self.attn_dim, dim=-1)
        attn_scores = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        attn_scores.masked_fill_(self.mask.bool()[:x.size(1), :x.size(1)], -torch.inf)
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        context_vec = self.dropout(attn_weights) @ v
        return context_vec, attn_scores

class MyStackedFunction(nn.Module):
    def __init__(self, module_cls, N: int, d_in: int, attn_dim: int, context_length: int, dropout: float = 0.1, qkv_bias: bool = False):
        super().__init__()
        self.linear = None
        self.N = N
        self.q = nn.Linear(d_in, attn_dim, bias=qkv_bias)
        self.k = nn.Linear(d_in, attn_dim, bias=qkv_bias)
        self.v = nn.Linear(d_in, attn_dim, bias=qkv_bias)
        self.modules_list = nn.ModuleList([
            module_cls(d_in, attn_dim, context_length, dropout=dropout, qkv_bias=qkv_bias)
            for _ in range(N)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs_z, outputs_weights = zip(*[mod(x) for mod in self.modules_list])
        stacked_z = torch.stack(outputs_z, dim=1)
        stacked_weights = torch.stack(outputs_weights, dim=1)
        B = stacked_z.shape[0]
        stacked_z_flat = stacked_z.flatten(start_dim=1)
        if self.linear is None:
            in_features = stacked_z_flat.shape[1]
            out_features = stacked_z.shape[-2] * stacked_z.shape[-1]
            self.linear = nn.Linear(in_features, out_features)
        y = self.linear(stacked_z_flat)
        y = y.reshape(stacked_z.shape[0], 1, stacked_z.shape[-2], stacked_z.shape[-1])
        return y, stacked_weights
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs_z, outputs_weights = zip(*[mod(x) for mod in self.modules_list])
        stacked_z = torch.stack(outputs_z, dim=1)  # shape: (B, N, ...)
        stacked_weights = torch.stack(outputs_weights, dim=1)  # shape: (B, N, ...)

        B = stacked_z.shape[0]
        # Flatten across N and remaining dims except batch
        stacked_z_flat = stacked_z.flatten(start_dim=1)  # shape: (B, N * ...)

        if self.linear is None:
            in_features = stacked_z_flat.shape[1]
            out_features = stacked_z.shape[-2] * stacked_z.shape[-1]  # same formula as your my_linear
            self.linear = nn.Linear(in_features, out_features)

        y = self.linear(stacked_z_flat)  # shape: (B, out_features)
        y = y.reshape(stacked_z.shape[0], 1, stacked_z.shape[-2], stacked_z.shape[-1])
        # Optionally reshape y to (B, 1, S, H) if needed
        return y, stacked_weights


class DummyTransformerBlock(nn.Module):  # 3
    def __init__(self, cfg):
        super().__init__()
        pass

class DummyLayerNorm(nn.Module):  # 5
    def __init__(self, normalized_shape, eps=1e-5):  # 6
        super().__init__()

    def forward(self, x):
        return x

def visualize_attention(tokens: list, attn_weights: torch.Tensor, head_idx: int):
    attn_weights = attn_weights.cpu().detach().numpy()
    print(f"\n--- Attention Head #{head_idx} ---")

    col_width = max(len(token) for token in tokens if token) + 2
    header = f"{'':<{col_width}}" + "".join([f"{token:^{col_width}}" for token in tokens])
    print(header)
    print("─" * len(header))

    for i, row_token in enumerate(tokens):
        row_str = f"{row_token:<{col_width}}"
        for j in range(len(tokens)):
            score = attn_weights[i, j]
            row_str += f"{score:^{col_width}.2f}"
        print(row_str)

def batch_encode(phrases, tokenizer, max_length):
    batch_tokens = [
        (tokens := tokenizer.encode(phrase))[:max_length] + [0] * (max_length - len(tokens))
        for phrase in phrases
    ]
    return torch.tensor(batch_tokens, dtype=torch.long)

h_dim = GPT_CONFIG_124M['h_dim']
attn_dim = GPT_CONFIG_124M['attn_dim']
num_heads = GPT_CONFIG_124M['num_heads']
max_length = GPT_CONFIG_124M['context_length']
vocab_path = os.path.join(os.path.dirname(__file__), 'byte_training', 'tok.json')
token_params = GPT_CONFIG_124M['tokenizer']
qkv_bias = token_params.get('qkv_bias', False)
#tokenizer = BytePairTokenizer()
tokenizer = TikToken200k


#vocab_size = max(tokenizer.vocab.values()) + 2
vocab_size = tokenizer.n_vocab
embedding_layer = nn.Embedding(vocab_size, h_dim, padding_idx=0)
pos_encoding_layer = PositionalEncoding(max_length, h_dim)
encoding_layer = Encoding(embedding_layer, pos_encoding_layer)
stacked_attention = MyStackedFunction(
    MaskedSelfAttention, num_heads, h_dim, attn_dim, max_length, dropout=0.1, qkv_bias=qkv_bias)

class attention_block(nn.Module):
    def __init__(self, num_heads: int, h_dim: int, attn_dim: int, max_length: int, dropout: float = 0.1, qkv_bias: bool = False):
        super().__init__()
        self.attention_and_linear = MyStackedFunction(
            MaskedSelfAttention, num_heads, h_dim, attn_dim, max_length, dropout=dropout, qkv_bias=qkv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention_and_linear(x)

if __name__ == '__main__':
    # only test if a "test=True" argument is passed, otherwise do nothing. Use model.test to run the tests.
    print("Running batch size validation test...")
    phrases = ['to pimp a butterfly', 'keep it moving forward']  # Only 2 phrases to match batch_size=2
    input_ids = batch_encode(phrases, tokenizer, max_length)  # (batch_size, seq_len)

    assert input_ids.shape[0] == len(phrases) == 2  # batch_size first
    assert input_ids.shape[1] == max_length

    encoded_sequences = encoding_layer(input_ids)  # (batch_size, seq_len, h_dim)
    assert encoded_sequences.shape[0] == 2
    assert encoded_sequences.shape[1] == max_length

    context, weights = stacked_attention(encoded_sequences)
    # context: (batch_size, 1, seq_len, attn_dim)
    # weights: (batch_size, num_heads, seq_len, seq_len)
    assert context.shape == (2, 1, max_length, attn_dim)
    assert weights.shape == (2, num_heads, max_length, max_length)
    print("✅ Batch size validation test passed.")
    print("-" * 20)

    print("\nRunning attention visualization test...")
    phrase = ['Abraham Lincoln', 'George Washington']  # batch_size=2
    input_tensor = batch_encode(phrase, tokenizer, max_length)  # (2, seq_len)
    assert input_tensor.shape[0] == 2
    encoded_single_sequence = encoding_layer(input_tensor)  # (2, seq_len, h_dim)
    assert encoded_single_sequence.shape[0] == 2
    context_1, weights_1 = stacked_attention(encoded_single_sequence)
    assert context_1.shape[0] == 2
    assert weights_1.shape[0] == 2

    for i in range(2):
        token_ids = input_tensor[i].tolist()
        string_tokens = [tokenizer.decode([j]) if j != 0 else "[PAD]" for j in token_ids]
        head_to_show = 2
        single_head_weights = weights_1[i, head_to_show, :, :]
        print(f"\nPhrase {i+1} visualization:")
        visualize_attention(
            tokens=string_tokens,
            attn_weights=single_head_weights,
            head_idx=head_to_show
        )
    print("✅ Attention visualization test finished.")

    # Single example, ensure batch dim is first
    phrase = 'Abraham Lincoln'
    ids = tokenizer.encode(phrase)
    while len(ids) < max_length:
        ids.append(50258)
    emb = encoding_layer(torch.tensor(ids).reshape(1, -1))  # (1, seq_len)
    assert emb.shape[0] == 1
    context = stacked_attention(emb)[0]