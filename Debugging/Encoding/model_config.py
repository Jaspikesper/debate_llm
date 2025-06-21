GPT_CONFIG_124M = {

"vocab_size": 10000,
    "context_length": 7,
    "h_dim": 128,
    "attn_dim": 128,
    "num_heads": 4,
    "num_layers": 4,
    "dropout_rate": 0.1,
    "activation_function": "gelu",
    "attn_bias": False,
    "tokenizer": {
        "type": "BytePairTokenizer",
        "target_vocab_size": 100000,
        "vocab_file": "tok.json",
    }

}