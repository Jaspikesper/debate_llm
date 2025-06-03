import regex
import os
import json
from collections import defaultdict, Counter
import tiktoken


class BytePairTokenizer:
    def __init__(self, target_vocab_size=None, vocab_file=None):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.byte_training_dir = os.path.join(BASE_DIR, 'byte_training')

        # Set default vocab file path if not provided
        if vocab_file is None:
            vocab_file = os.path.join(self.byte_training_dir, 'tok.json')
        self.vocab_file = vocab_file

        corpus_parts = []
        for trainfile in os.listdir(self.byte_training_dir):
            trainfile_path = os.path.join(self.byte_training_dir, trainfile)
            with open(trainfile_path, 'r', encoding='utf-8') as f:
                corpus_parts.append(f.read())
                corpus_parts.append('<|endoftext|>')
        self.training_corpus = ''.join(corpus_parts)
        self.training_corpus_preserved = str(self.training_corpus)

        # Initialize vocab with special tokens
        self.vocab = {
            '<|endoftext|>': 50256,
            '<|unknown|>': 50257,  # Added unknown token
        }
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.target_vocab_size = target_vocab_size

        # Pattern to tokenize similar to GPT-2
        self.gpt2pattern = regex.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        # Build initial vocabulary from corpus
        self.update_vocab(self.training_corpus)
        self.sorted_keys = sorted(self.vocab.keys(), key=lambda x: -len(x))
        self._build_prefix_index()

        # Load pre-existing vocabulary if available
        print(self.vocab_file)
        if os.path.exists(self.vocab_file):
            self.load(self.vocab_file)
        else:
            self.update_vocab(self.training_corpus)

        if self.target_vocab_size and len(self.vocab) < self.target_vocab_size:
            self.train(self.training_corpus)

        self.vocab_size = len(self.vocab)

    # Build prefix index for efficient token lookup
    def _build_prefix_index(self):
        self.prefix_index = defaultdict(list)
        for tok in self.vocab:
            self.prefix_index[tok[0]].append(tok)
        for lst in self.prefix_index.values():
            lst.sort(key=len, reverse=True)

    # Update vocabulary with new tokens
    def update_vocab(self, tokens):
        added = False
        for tok in tokens:
            if isinstance(tok, str) and len(tok) == 1:
                if tok not in self.vocab:
                    self.vocab[tok] = ord(tok)
                    added = True
            elif tok not in self.vocab:
                self.vocab[tok] = tok
                added = True
        if added:
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}
            self.sorted_keys = sorted(self.vocab.keys(), key=lambda x: -len(x))
            self._build_prefix_index()

    # Encode a string sequence into token ids
    def encode(self, seq):
        output_ids = []
        seq_copy = seq
        while seq_copy:
            matched = False
            for key in self.prefix_index.get(seq_copy[0], self.sorted_keys):
                if seq_copy.startswith(key):
                    output_ids.append(self.vocab[key])
                    seq_copy = seq_copy[len(key):]
                    matched = True
                    break
            if not matched:
                # Use unknown token if no known prefix found
                output_ids.append(self.vocab['<|unknown|>'])
                # Remove one grapheme (character) to avoid infinite loop
                first_grapheme = regex.findall(r'\X', seq_copy)[0]
                seq_copy = seq_copy[len(first_grapheme):]
        return output_ids

    # Decode token ids back into a string
    def decode(self, ids):
        return ''.join(self.inverse_vocab.get(i, '<|unknown|>') for i in ids)

    # Segment a sequence according to GPT-2 style tokenization
    def segment(self, seq):
        seq_decoded = self.decode(seq)
        segmented = regex.findall(self.gpt2pattern, seq_decoded)
        return [self.encode(word) for word in segmented]

    # Identify the most frequent pair of tokens for merging
    def get_frequencies(self, segs):
        pair_frequency = defaultdict(int)
        for seg in segs:
            for x, y in zip(seg, seg[1:]):
                pair_frequency[(x, y)] += 1
        if not pair_frequency:
            return None
        return max(pair_frequency.items(), key=lambda item: item[1])[0]

    # Merge the most frequent token pair and update vocab
    def merge_and_update(self, ids, pair, idx):
        translation = self.inverse_vocab[pair[0]] + self.inverse_vocab[pair[1]]
        new_ids = []
        i = 0
        n = len(ids)
        while i < n:
            if i < n - 1 and (ids[i], ids[i + 1]) == pair:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        self.vocab[translation] = idx
        self.inverse_vocab[idx] = translation
        self.sorted_keys = sorted(self.vocab.keys(), key=lambda x: -len(x))
        self._build_prefix_index()
        return new_ids

    # Train tokenizer iteratively
    def train(self, seq):
        ids = self.encode(seq)
        while len(self.vocab) < self.vocab_size:
            segs = self.segment(ids)
            top_pair = self.get_frequencies(segs)
            if top_pair is None:
                break
            new_id = max(self.vocab.values()) + 1
            ids = self.merge_and_update(ids, top_pair, new_id)
            if len(self.vocab) % 5 == 0:
                print("Saving current dictionary of length: ", len(self.vocab))
                self.save()
        self.save()

    # Save vocab to file
    def save(self, fstring='Encoding/byte_training/tok.json'):
        if fstring == 'Encoding/byte_training/tok.json':
            fstring = self.vocab_file

        # Ensure the directory exists
        os.makedirs(os.path.dirname(fstring), exist_ok=True)

        with open(fstring, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f)

    # Load vocab from file
    def load(self, fstring='Encoding/byte_training/tok.json'):
        if fstring == 'Encoding/byte_training/tok.json':
            fstring = self.vocab_file

        with open(fstring, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        print('loaded a model with ', len(self.vocab), 'vocabulary!')
        print('target vocab size is: ', self.target_vocab_size, '!')
        self.inverse_vocab = {int(v): k for k, v in self.vocab.items()}
        self.sorted_keys = sorted(self.vocab.keys(), key=lambda x: -len(x))
        self._build_prefix_index()


TikToken200k = tiktoken.get_encoding("o200k_base")

TikToken4o = tiktoken.encoding_for_model("gpt-4o")

if __name__ == '__main__':
    tok = BytePairTokenizer(target_vocab_size=32000, vocab_file=r'C:\Users\jaspe\PycharmProjects\PythonProject7\Encoding\byte_training\tok.json')
