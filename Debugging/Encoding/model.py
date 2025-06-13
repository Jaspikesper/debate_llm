import torch
import torch.nn as nn
import math
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tokenizer import BytePairTokenizer


# --- Model Component Classes ---

class PositionalEncoding(nn.Module):
    def __init__(self, max_length, h_dim):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_length, h_dim)

    def forward(self, x):
        seq_length = x.size(1)
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0).expand_as(x)
        return self.pos_embedding(positions)


class Encoding(nn.Module):
    def __init__(self, embedding_layer, pos_encoding):
        super().__init__()
        self.embedding = embedding_layer
        self.pos_encoding = pos_encoding

    def forward(self, x):
        embeds = self.embedding(x)
        pos_embeds = self.pos_encoding(x)
        return embeds + pos_embeds


class MaskedSelfAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)  # #1
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )  # #2

    def forward(self, x):
        b, num_tokens, d_in = x.shape  # #3
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )  # #4

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec, attn_scores


class MyStackedFunction(nn.Module):
    """
    Correctly stacks multiple MaskedSelfAttention modules to create a multi-head attention block.
    This version fixes the previous math errors that produced NaN values.
    """

    def __init__(self, module_cls, N, *args, **kwargs):
        super().__init__()
        self.modules_list = nn.ModuleList([module_cls(*args, **kwargs) for _ in range(N)])

    def forward(self, x):
        # Each module returns a tuple of (z, weights)
        outputs = [mod(x) for mod in self.modules_list]

        # Unzip the list of (z, weights) tuples into two separate lists
        outputs_z, outputs_weights = zip(*outputs)

        # Stack along a new dimension (dim=1) to represent the heads.
        # This is the correct, mathematically stable way to aggregate the heads.
        # Shape: (batch, num_heads, seq, d_out)
        stacked_z = torch.stack(outputs_z, dim=1)
        # Shape: (batch, num_heads, seq, seq)
        stacked_weights = torch.stack(outputs_weights, dim=1)

        # Always return both tensors. The calling code can decide to ignore weights.
        return stacked_z, stacked_weights


# --- Visualization and Testing ---

def visualize_attention(tokens: list[str], attn_weights: torch.Tensor, head_idx: int):
    """Helper function to print a formatted grid of attention weights."""
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


def create_token_tensor(phrases, tokenizer, max_length):
    """Helper to tokenize and pad a list of phrases."""
    batch_tokens = []
    for phrase in phrases:
        tokens = tokenizer.encode(phrase)
        if len(tokens) < max_length:
            tokens += [0] * (max_length - len(tokens))  # Pad with 0
        else:
            tokens = tokens[:max_length]
        batch_tokens.append(tokens)
    return torch.tensor(batch_tokens, dtype=torch.long)

if __name__ == '__main__':
    # 1. --- Setup Hyperparameters ---
    h_dim = 18
    max_length = 3
    num_heads = 4
    d_out = 16

    # --- Initialize Tokenizer ---
    try:
        vocab_path = os.path.join(os.path.dirname(__file__), 'byte_training', 'tok.json')
        tokenizer = BytePairTokenizer(vocab_file=vocab_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Please ensure 'tok.json' file is in the 'byte_training' directory.")
        exit()

    vocab_size = max(tokenizer.vocab.values()) + 2

    # --- Setup Model Components ---
    embedding_layer = nn.Embedding(vocab_size, h_dim, padding_idx=0)
    pos_encoding_layer = PositionalEncoding(max_length, h_dim)
    encoding_layer = Encoding(embedding_layer, pos_encoding_layer)
    stacked_attention = MyStackedFunction(
        MaskedSelfAttention, num_heads, h_dim, d_out, max_length, dropout=0.0
    )

    # --- Test Batch Size Validation ---
    print("Running batch size validation test...")
    phrases = ['to pimp a butterfly', 'keep it moving forward', 'it is okay to be gay', 'money is not everything']
    input_ids = create_token_tensor(phrases, tokenizer, max_length)

    # Encode input sequences using the Encoding layer
    encoded_sequences = encoding_layer(input_ids)

    # Pass through the stacked attention layers
    context, weights = stacked_attention(encoded_sequences)

    # All outputs should be valid tensors now
    assert context is not None and weights is not None
    assert context.shape == (len(phrases), num_heads, max_length, d_out)
    assert weights.shape == (len(phrases), num_heads, max_length, max_length)
    print("✅ Batch size validation test passed.")
    print("-" * 20)

    # --- Test and Visualize Single Instance ---
    print("\nRunning attention visualization test...")
    phrase = ['Abraham Lincoln']  # Your favorite phrase
    input_tensor = create_token_tensor(phrase, tokenizer, max_length)

    # Encode single input sequence
    encoded_single_sequence = encoding_layer(input_tensor)

    # Pass through the stacked attention layers
    context_1, weights_1 = stacked_attention(encoded_single_sequence)

    # Decode for visualization
    token_ids = input_tensor[0].tolist()
    string_tokens = [tokenizer.decode([i]) if i != 0 else "[PAD]" for i in token_ids]

    # Select a head and visualize
    head_to_show = 2
    single_head_weights = weights_1[0, head_to_show, :, :]

    visualize_attention(
        tokens=string_tokens,
        attn_weights=single_head_weights,
        head_idx=head_to_show
    )
    print("✅ Attention visualization test finished.")
