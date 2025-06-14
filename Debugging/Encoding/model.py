import os
import sys
import torch
import torch.nn as nn
import math
from tokenizer import BytePairTokenizer

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
    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float = 0.1, qkv_bias: bool = False):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.d_out = d_out
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)
        attn_scores = queries @ keys.transpose(-2, -1) / math.sqrt(keys.size(-1))
        attn_scores.masked_fill_(self.mask.bool()[:x.size(1), :x.size(1)], -torch.inf)
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        context_vec = self.dropout(attn_weights) @ values
        return context_vec, attn_scores


class MyStackedFunction(nn.Module):
    def __init__(self, module_cls, N: int, *args, **kwargs):
        super().__init__()
        self.modules_list = nn.ModuleList([module_cls(*args, **kwargs) for _ in range(N)])
        self.linear = None
        self.N = N

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


class MultiHeadedAttention(nn.Module):
    #1: initialize a list of MaskedSelfAttention modules
    #2: Take the for each of Q K and V, stack the respective matrices along the batch dimension
    #3: perform multi-headed attention on the batched input to get stacked context
    pass

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
    print()

def create_token_tensor(phrases: list, tokenizer: BytePairTokenizer, max_length: int) -> torch.Tensor:
    batch_tokens = []
    for phrase in phrases:
        tokens = tokenizer.encode(phrase)
        tokens = tokens[:max_length] if len(tokens) > max_length else tokens + [0] * (max_length - len(tokens))
        batch_tokens.append(tokens)
    return torch.tensor(batch_tokens, dtype=torch.long)

if __name__ == '__main__':
    h_dim = 18
    max_length = 3
    num_heads = 7
    d_out = 16

    vocab_path = os.path.join(os.path.dirname(__file__), 'byte_training', 'tok.json')
    tokenizer = BytePairTokenizer(vocab_file=vocab_path)

    vocab_size = max(tokenizer.vocab.values()) + 2
    embedding_layer = nn.Embedding(vocab_size, h_dim, padding_idx=0)
    pos_encoding_layer = PositionalEncoding(max_length, h_dim)
    encoding_layer = Encoding(embedding_layer, pos_encoding_layer)
    stacked_attention = MyStackedFunction(
        MaskedSelfAttention, num_heads, h_dim, d_out, max_length, dropout=0.1)

    print("Running batch size validation test...")
    phrases = ['to pimp a butterfly', 'keep it moving forward', 'it is okay to be gay', 'money is not everything']
    input_ids = create_token_tensor(phrases, tokenizer, max_length)

    encoded_sequences = encoding_layer(input_ids)
    context, weights = stacked_attention(encoded_sequences)

    assert context.shape == (len(phrases), 1, max_length, d_out)
    assert weights.shape == (len(phrases), num_heads, max_length, max_length)
    print("✅ Batch size validation test passed.")
    print("-" * 20)

    print("\nRunning attention visualization test...")
    phrase = ['Abraham Lincoln']
    input_tensor = create_token_tensor(phrase, tokenizer, max_length)
    encoded_single_sequence = encoding_layer(input_tensor)
    context_1, weights_1 = stacked_attention(encoded_single_sequence)

    token_ids = input_tensor[0].tolist()
    string_tokens = [tokenizer.decode([i]) if i != 0 else "[PAD]" for i in token_ids]

    head_to_show = 2
    single_head_weights = weights_1[0, head_to_show, :, :]

    visualize_attention(
        tokens=string_tokens,
        attn_weights=single_head_weights,
        head_idx=head_to_show
    )
    print("✅ Attention visualization test finished.")

    phrase = 'Abraham Lincoln'
    ids = tokenizer.encode(phrase)
    while len(ids) < max_length:
        ids.append(50258)
    emb = encoding_layer(torch.tensor(ids).reshape(1, -1))
    context = stacked_attention(emb)[0]
