import regex
import os
from Encoding.embedding import vocab_size
import json

class BytePairTokenizer:
    def __init__(self, vocab_size=512, pretrained_file='tok.json'):
        self.training_corpus = ''
        self.vocab = {'<|endoftext|>': 50256}  # token -> id (character -> UTF-8 code point)
        for trainfile in os.listdir('byte_training'):
            trainfile = os.path.join('byte_training', trainfile)
            self.training_corpus += open(trainfile, 'r').read()
            self.training_corpus += '<|endoftext|>'
        self.training_corpus_preserved = str(self.training_corpus)
        self.inverse_vocab = {50256: '<|endoftext|>'}  # id -> token
        self.vocab_size = vocab_size
        self.update_vocab(self.training_corpus)
        self.gpt2pattern = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        if pretrained_file and os.path.exists(pretrained_file):
            self.load(pretrained_file)
        else:
            self.train(self.training_corpus)
        self.segs = []

    def segment(self, seq, count):
        seq = self.decode(seq)
        segmented = regex.findall(self.gpt2pattern, seq)
        ids = self.encode(seq)
        segmented_seq = [self.encode(word) for word in segmented]
        return segmented_seq


    def update_vocab(self, tokens):
        for tok in tokens:
            if isinstance(tok, str) and len(tok) == 1:
                self.vocab[tok] = ord(tok)
            else:
                self.vocab[tok] = tok
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, seq):
        tokens = regex.findall(self.gpt2pattern, seq)
        output_ids = []

        for token in tokens:
            chars = list(token)
            i = 0
            while i < len(chars):
                # Try to find the longest possible token from position i
                j = len(chars)
                found = False
                while j > i:
                    candidate = ''.join(chars[i:j])
                    if candidate in self.vocab:
                        output_ids.append(self.vocab[candidate])
                        i = j
                        found = True
                        break
                    j -= 1
                if not found:
                    # Fallback to char-level vocab (assumes it's there)
                    output_ids.append(self.vocab.get(chars[i], ord(chars[i])))
                    i += 1
        return output_ids

    def decode(self, ids):
        return ''.join([self.inverse_vocab[i] for i in ids])

    def get_frequencies(self, segs):
        pair_frequency = {}
        for seg in segs:
            for x, y in zip(seg, seg[1:]):
                pair_frequency[(x, y)] = pair_frequency.get((x, y), 0) + 1
        return max(pair_frequency.items(), key=lambda item: item[1])[0]

    def merge_and_update(self, ids, pair, idx):
        translation = self.inverse_vocab[pair[0]] + self.inverse_vocab[pair[1]]
        obsolete_locs = []
        for i in range(len(ids) - 1):
            if (ids[i], ids[i+1]) == pair:
                ids[i] = idx
                obsolete_locs.append(i + 1)
        ids = [id_ for j, id_ in enumerate(ids) if j not in obsolete_locs]
        self.vocab[translation] = idx
        self.inverse_vocab[idx] = translation
        print('inverse vocab at: ', idx, 'is: ', translation)
        return ids

    def train(self, seq=None):
        if seq is None:
            seq = self.training_corpus
        ids = self.encode(seq)
        while len(self.vocab) < self.vocab_size:
            segs = self.segment(ids, count)
            top_pair = self.get_frequencies(segs)
            new_id = max(self.vocab.values()) + 1
            print(f'Merging the token subsequence {top_pair[0]}, {top_pair[1]} into token ID {new_id}')
            print(f'Merging the subsequence {self.inverse_vocab[top_pair[0]]}, {self.inverse_vocab[top_pair[1]]} into token ID {new_id}')
            ids = self.merge_and_update(ids, top_pair, new_id)
            self.training_corpus = ids
        print('Length of sequence before encoding (character sequence):', len(self.training_corpus_preserved))
        print('Length of encoded sequence (byte-pair token sequence):', len(ids))
        compression_ratio = len(self.training_corpus_preserved) / len(ids)
        print('Compression ratio:', compression_ratio)
        self.save()

    def save(self, fstring='tok.json'):
        with open(fstring, 'w', encoding='utf-8') as file:
            json.dump(self.vocab, file)

    def load(self, fstring='tok.json'):
        with open(fstring, 'r', encoding='utf-8') as file:
            self.vocab = json.load(file)
        self.inverse_vocab = {int(v): k for k, v in self.vocab.items()}

if __name__ == '__main__':
    tok = BytePairTokenizer(vocab_size=300, pretrained_file=None)
