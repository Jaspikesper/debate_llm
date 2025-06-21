import torch
from model import *


if __name__ == '__main__':
    phrase = 'Abraham Lincoln'
    tokens = tokenizer.encode(phrase) # Tokenizer is predefined in model.py
    x = encoding_layer(torch.tensor(tokens, dtype=torch.long).unsqueeze(0))
    context, _ = stacked_attention(x)