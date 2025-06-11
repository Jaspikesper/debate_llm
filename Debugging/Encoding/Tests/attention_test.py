import sys
import os

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODING_DIR = os.path.abspath(os.path.join(CURR_DIR, '..'))
if ENCODING_DIR not in sys.path:
    sys.path.insert(0, ENCODING_DIR)

import torch
from tokenizer import BytePairTokenizer
from data_loader import GPT_Dataset, make_dataloader

# Import all model classes and visualize_attention from model.py
from model import PositionalEncoding, Encoding, MaskedSelfAttention, MyStackedFunction, visualize_attention

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run attention module tests.")
    parser.add_argument("--visualize", action="store_true", help="Run the attention visualization test.")
    args = parser.parse_args()

    h_dim = 18
    max_length = 12
    num_heads = 4
    d_out = 16
    batch_size = 6

    # Find a training or test data file (adjust as needed)
    data_file = os.path.join(ENCODING_DIR, 'byte_training', 'LLM_dataset.txt')
    if not os.path.exists(data_file):
        # Fallback for a different possible file name (adjust if your corpus file is different)
        for fname in os.listdir(os.path.join(ENCODING_DIR, 'byte_training')):
            if fname.endswith('.txt'):
                data_file = os.path.join(ENCODING_DIR, 'byte_training', fname)
                break

    with open(data_file, "r", encoding="utf-8") as f:
        text_data = f.read()

    tokenizer = BytePairTokenizer(vocab_file=os.path.join(ENCODING_DIR, 'byte_training', 'tok.json'))
    vocab_size = max(tokenizer.vocab.values()) + 2

    dataset = GPT_Dataset(text_data, tokenizer, max_length=max_length, stride=5)

    embedding_layer = torch.nn.Embedding(vocab_size, h_dim, padding_idx=0)
    pos_encoding_layer = PositionalEncoding(max_length, h_dim)
    encoding_layer = Encoding(embedding_layer, pos_encoding_layer)
    stacked_attention = MyStackedFunction(MaskedSelfAttention, num_heads, h_dim, d_out, max_length, dropout=0.1)

    # Batch size test
    dataloader_N = make_dataloader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    dataloader_1 = make_dataloader(dataset, batch_size=1, shuffle=True)

    # Test batch > 1
    input_tensor_N, _ = next(iter(dataloader_N))
    embedded_N = encoding_layer(input_tensor_N)
    context_N, weights_N = stacked_attention(embedded_N)
    assert weights_N is None, "Weights should be None for batch_size > 1"
    assert context_N.shape == (batch_size, num_heads, max_length, d_out)

    # Test batch == 1
    input_tensor_1, _ = next(iter(dataloader_1))
    embedded_1 = encoding_layer(input_tensor_1)
    context_1, weights_1 = stacked_attention(embedded_1)
    assert weights_1 is not None, "Weights should be returned for batch_size=1"
    assert context_1.shape == (1, num_heads, max_length, d_out)
    assert weights_1.shape == (1, num_heads, max_length, max_length)

    print("✅ Batch size validation test passed.")
    print("-" * 20)

    # Optional: Visualization
    if args.visualize:
        tokens = [tokenizer.decode([i.item()]) for i in input_tensor_1[0]]
        head_to_show = 0
        single_head_weights = weights_1[0, head_to_show, :, :]
        visualize_attention(tokens, single_head_weights, head_idx=head_to_show)
        print("✅ Attention visualization test finished.")

if __name__ == '__main__':
    main()