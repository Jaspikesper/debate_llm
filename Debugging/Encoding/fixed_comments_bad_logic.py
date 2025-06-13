import torch
from model import *
from tokenizer import BytePairTokenizer, TikToken4o
import math
from data_loader import GPT_Dataset, make_dataloader  # Note: Correct capitalization here

# Data hyperparameters
h_dim = 18
max_length = 6

# Initialize tokenizer and embedding layer
tokenizer = BytePairTokenizer(vocab_file='byte_training/tok.json')
vocab_size = max(tokenizer.vocab.values()) + 2
embedding_layer = torch.nn.Embedding(vocab_size, h_dim, padding_idx=0)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, max_length, h_dim):
        super().__init__()
        self.pos_embedding = torch.nn.Embedding(max_length, h_dim)

    def forward(self, x):
        # x: (batch_size, seq_length)
        seq_length = x.size(1)
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0).expand_as(x)
        return self.pos_embedding(positions)


# Encoding Module should work with every conceivable batched input size due to its use of nn.linear layers.
class Encoding(torch.nn.Module):
    def __init__(self, embedding_layer, pos_encoding):
        super().__init__()
        self.embedding = embedding_layer
        self.pos_encoding = pos_encoding

    def forward(self, x):
        embeds = self.embedding(x)
        pos_embeds = self.pos_encoding(x)
        return embeds + pos_embeds


# Single instance of masked and scaled dot product self-attention. Can be run in parallel with separate instances.
class MaskedSelfAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout=0.5, qkv_bias=True):
        super().__init__()
        self.W_q = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.scale = 1 / math.sqrt(d_out)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1) * float('-inf'))

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


class MyStackedFunction(torch.nn.Module):
    def __init__(self, module_cls, N, *args, **kwargs):
        super().__init__()
        self.modules_list = torch.nn.ModuleList([module_cls(*args, **kwargs) for _ in range(N)])

    def forward(self, x):
        # x: (batch_size, seq_length, d_in)
        outputs = [mod(x) for mod in self.modules_list]  # list of (batch_size, seq_length, d_out)
        # Stack to (num_heads, batch_size, seq_length, d_out)
        outputs = torch.stack(outputs, dim=0)
        # Rearrange to (batch_size, num_heads, seq_length, d_out)
        outputs = outputs.permute(1, 0, 2, 3)
        return outputs


num_heads = 12
d_out = 16  # Required before using in StackedAttention

# Should work correctly according to function signature of attention and the stacking thereof
StackedAttention = MyStackedFunction(
    MaskedSelfAttention,
    num_heads,
    h_dim,
    d_out,
    max_length,
    dropout=0.5,
    qkv_bias=True
)

if __name__ == '__main__':
    phrases = ['to pimp a ', 'keep it moving', 'its ok to be gay', 'money aint everything']

    batch_tokens = []
    for phrase in phrases:
        tokens = tokenizer.encode(phrase)
        # Pad or trim to max_length
        if len(tokens) < max_length:
            tokens += [0] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
        batch_tokens.append(tokens)
    input_tensor = torch.tensor(batch_tokens, dtype=torch.long)  # (batch_size=4, seq_length=max_length)

    pos_encoding_layer = PositionalEncoding(max_length=max_length, h_dim=h_dim)
    encoding_layer = Encoding(embedding_layer, pos_encoding_layer)
    attention_layer = MaskedSelfAttention(d_in=h_dim, d_out=d_out, context_length=max_length)

    embedded_with_pos = encoding_layer(input_tensor)  # (4, seq_length, h_dim)
    context = StackedAttention(embedded_with_pos)  # (4, num_heads, seq_length, d_out)

    print(context.shape)  # should be (4, num_heads, max_length, d_out)


