import sys
import os
import torch

# Add parent directory to sys.path if running from tests/ or root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

# Now safe to import from the main codebase
from tokenizer import BytePairTokenizer
from model import PositionalEncoding, Encoding, MaskedSelfAttention, MyStackedFunction

def test_attention_shapes():
    # Model/test hyperparameters
    h_dim = 18
    max_length = 6
    num_heads = 12
    d_out = 16

    # Setup
    vocab_file = os.path.join(PARENT_DIR, "byte_training", "tok.json")
    tokenizer = BytePairTokenizer(vocab_file=vocab_file if os.path.exists(vocab_file) else None)
    vocab_size = max(tokenizer.vocab.values()) + 2
    embedding_layer = torch.nn.Embedding(vocab_size, h_dim, padding_idx=0)
    pos_encoding_layer = PositionalEncoding(max_length=max_length, h_dim=h_dim)
    encoding_layer = Encoding(embedding_layer, pos_encoding_layer)
    stacked_attention = MyStackedFunction(
        MaskedSelfAttention, num_heads, h_dim, d_out, max_length, dropout=0.5, qkv_bias=True
    )

    # Prepare batch
    phrases = [
        'to pimp a ',
        'keep it moving',
        'its ok to be gay',
        'money aint everything'
    ]
    batch_tokens = []
    for phrase in phrases:
        tokens = tokenizer.encode(phrase)
        tokens = tokens[:max_length] + [0] * (max_length - len(tokens)) if len(tokens) < max_length else tokens[:max_length]
        batch_tokens.append(tokens)
    input_tensor = torch.tensor(batch_tokens, dtype=torch.long)  # (batch_size=4, seq_length=max_length)

    # Embedding and attention
    embedded_with_pos = encoding_layer(input_tensor)  # (batch, seq_len, h_dim)
    context = stacked_attention(embedded_with_pos)    # (batch, num_heads, seq_len, d_out)

    print("Context tensor shape:", context.shape)
    assert context.shape == (4, num_heads, max_length, d_out), "Shape mismatch in attention output!"

if __name__ == '__main__':
    test_attention_shapes()
    print("Attention stacking/shapes test passed.")