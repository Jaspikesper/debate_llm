import torch
import torch.nn as nn
import math
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tokenizer import BytePairTokenizer


# Initialize PositionalEncoding with specified max sequence length and hidden dimension
class PositionalEncoding(nn.Module):
    # Establish learnable positional encoding parameters
    def __init__(self, max_length, h_dim):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_length, h_dim)

    def forward(self, x):
        # Determine the length of the incoming sequence
        seq_length = x.size(1)
        # Generate position indices and return corresponding embeddings
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0).expand_as(x)
        return self.pos_embedding(positions)


# Encoding combines both token embeddings and positional encoding for input
class Encoding(nn.Module):
    # Encoding Module should work with every conceivable batched input size due to its use of nn.linear layers.
    # Initialize with embedding and positional encoding layers
    def __init__(self, embedding_layer, pos_encoding):
        super().__init__()
        self.embedding = embedding_layer
        self.pos_encoding = pos_encoding

    def forward(self, x):
        # Compute token embeddings and positional embeddings
        embeds = self.embedding(x)
        pos_embeds = self.pos_encoding(x)
        return embeds + pos_embeds


# Single instance of masked and scaled dot product self-attention. Can be run in parallel with separate instances.
# Implement masked self-attention with dropout and bias options
class MaskedSelfAttention(nn.Module):
    # Initialize attention mechanism with input and output dimensions
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        # Define linear projections for queries, keys, and values
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)  # #1

        # Create and register buffer for masking attention scores
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )  # #2

    def forward(self, x):
        # Extract batch size, sequence length, and input dimension
        b, num_tokens, d_in = x.shape  # #3

        # Compute projections for queries, keys, and values
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Calculate raw attention scores, applying scaling
        attn_scores = queries @ keys.transpose(1, 2)
        # Apply mask to ensure no attention to future tokens
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )  # #4

        # Normalize attention scores and apply dropout
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        # Compute context vectors from weighted values
        context_vec = attn_weights @ values
        return context_vec, attn_scores


# Stacking thereof
class MyStackedFunction(nn.Module):
    """
    A multi-head attention mechanism with stacked attention modules.
    This version fixes the previous math errors that produced NaN values.
    """

    # Stack multiple MaskedSelfAttention modules
    def __init__(self, module_cls, N, *args, **kwargs):
        super().__init__()
        self.modules_list = nn.ModuleList([module_cls(*args, **kwargs) for _ in range(N)])

    def forward(self, x):
        # Each module returns a tuple of (z, weights)
        outputs = [mod(x) for mod in self.modules_list]

        # Unzip the list of (z, weights) tuples into two separate lists
        outputs_z, outputs_weights = zip(*outputs)

        # Aggregate context vectors and attention weights across heads
        stacked_z = torch.stack(outputs_z, dim=1)  # (batch, num_heads, seq, d_out)
        stacked_weights = torch.stack(outputs_weights, dim=1)  # (batch, num_heads, seq, seq)

        # Return context vectors and attention weights for all heads
        return stacked_z, stacked_weights


# Function to print formatted attention weights for a single head
def visualize_attention(tokens: list[str], attn_weights: torch.Tensor, head_idx: int):
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


# Tokenize and pad a list of phrases to create a tensor
def create_token_tensor(phrases, tokenizer, max_length):
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
    # Set hyperparameters
    h_dim = 18
    max_length = 3
    num_heads = 4
    d_out = 16

    # Initialize Tokenizer
    try:
        vocab_path = os.path.join(os.path.dirname(__file__), 'byte_training', 'tok.json')
        tokenizer = BytePairTokenizer(vocab_file=vocab_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Please ensure 'tok.json' file is in the 'byte_training' directory.")
        exit()

    vocab_size = max(tokenizer.vocab.values()) + 2

    # Initialize Model Components
    embedding_layer = nn.Embedding(vocab_size, h_dim, padding_idx=0)
    pos_encoding_layer = PositionalEncoding(max_length, h_dim)
    encoding_layer = Encoding(embedding_layer, pos_encoding_layer)
    stacked_attention = MyStackedFunction(
        MaskedSelfAttention, num_heads, h_dim, d_out, max_length, dropout=0.0
    )

    print("Running batch size validation test...")
    phrases = ['to pimp a butterfly', 'keep it moving forward', 'it is okay to be gay', 'money is not everything']
    input_ids = create_token_tensor(phrases, tokenizer, max_length)

    # Forward pass through encoding and attention layers
    encoded_sequences = encoding_layer(input_ids)
    context, weights = stacked_attention(encoded_sequences)

    assert context is not None and weights is not None
    assert context.shape == (len(phrases), num_heads, max_length, d_out)
    assert weights.shape == (len(phrases), num_heads, max_length, max_length)
    print("✅ Batch size validation test passed.")
    print("-" * 20)

    # Run single instance visualization test
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