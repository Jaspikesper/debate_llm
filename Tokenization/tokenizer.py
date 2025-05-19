import re
import json
import os

class SimpleTokenizer:

    def __init__(self):
        self.vocab = {}  # str -> int
        self.inverse_vocab = {}  # int -> str
        self.n = 0


    def encode(self, text):
        # Include whitespace as its own token
        separated = re.findall(r'\s+|\w+|[^\w\s]', text)
        new_vocab = dict.fromkeys(separated)
        self.vocab.update(new_vocab)
        self.n += len(new_vocab)
        self.inverse_vocab = dict(zip(self.vocab.values(), self.vocab.keys()))
        return [self.vocab[token] for token in separated]

    def decode(self, ids):
        return ''.join([self.inverse_vocab[i] for i in ids])

class BytePairTokenizer:
    """ Byte pair tokenizer class contains all necessary objects for:
     1: Extending vocabulary to contain new characters/tokens
     2: Tracking the most common paired token occurrences
     3: At will, merging the most commonly paired tokens into one
     4: Encoding and decoding using all necessary tokens (both new and old)

     """
    def __init__(self, pretrained_file='tok.json'):
        self.training_corpus = open('byte_training.txt').read()
        self.training_corpus_preserved = str(self.training_corpus)
        self.vocab = {} # token -> id   e.g. {'re':99}
        self.inverse_vocab = {} # id -> token e.g. {99:'re'}
        self.vocab_size = 300
        self.update_vocab(self.training_corpus)
        if pretrained_file:
            self.load(pretrained_file)
        else:
            self.train(self.training_corpus)


    def update_vocab(self, tokens): # adds previously unseen character tokens to the model's vocabulary
        for ch in tokens:
            if ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)
        self.inverse_vocab.update({v:k for k, v in self.vocab.items()})

    def encode(self, seq): # and also update the vocab dict with the new vocab
        self.update_vocab(seq)
        ids = [self.vocab[tok] for tok in seq]
        return ids

    def decode(self, ids):
        return ''.join([self.inverse_vocab[id] for id in ids])

    def get_frequencies(self, ids):
        pair_frequency = {}
        for x, y in zip(ids, ids[1:]):
            try:
                pair_frequency[(x, y)] += 1
            except KeyError:
                pair_frequency[(x, y)] = 1
        top_pair = max(pair_frequency.items(), key=lambda item: item[1])[0] # the frequency of each pair
        return top_pair

    def merge_and_update(self, ids, pair, idx): # search and substitute a single token (idx) for a certain pair within a list of ids (NOT the vocab
        obsolete_locs = []
        translation = self.inverse_vocab[pair[0]] + self.inverse_vocab[pair[1]]
        for i in range(len(ids)):
            try:
                if (ids[i], ids[i+1]) == pair:
                    ids[i] = idx
                    obsolete_locs.append(i+1)
            except IndexError:
                pass
        ids = [ids[i] for i in range(len(ids)) if i not in obsolete_locs]
        self.vocab.update({idx: translation})
        self.inverse_vocab.update({idx: translation})
        return ids

    def train(self, seq):
        ids = self.encode(seq) # copy this?
        while len(self.vocab) < self.vocab_size:
            top_pair = self.get_frequencies(ids)
            new_id = len(self.vocab) + 1
            print(f'Merging the subsequence_{self.inverse_vocab[top_pair[0]], self.inverse_vocab[top_pair[1]]}_into token ID {new_id}')
            ids = self.merge_and_update(ids, top_pair, new_id)
            self.training_corpus = ids
        print('Length of sequence before encoding (character sequence): ', len(self.training_corpus_preserved))
        print('Length of encoded sequence (byte-pair token sequence): ', len(ids))
        compression_ratio = len(self.training_corpus_preserved) / len(ids)
        print('Compression ratio: ', compression_ratio)
        self.save()

    def save(self, fstring='tok.json'):
        with open(fstring, 'w') as file:
            json.dump(self.vocab, file)

    def load(self, fstring='tok.json'):
        with open(fstring, 'r') as file:
            self.vocab = json.load(file)