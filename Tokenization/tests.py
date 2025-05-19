from urllib.request import urlretrieve
import pandas as pd
from Tokenization.tokenizer import *
rawtext_link = 'https://www.gutenberg.org/ebooks/26184.txt.utf-8'
urlretrieve(rawtext_link, '../my_text.txt')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

class TokenTester:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.examples = [
            ''' To create a briefer but even hotter flame, 
            ''',
            '''Bushings and other exposed electrical parts.
            ''',
            '''(1.) Spoil fruit and vegetables by leaving them in the sun.
            ''',
            '''(1) Exchange wires in switchboards containing signals and switches,
            ''',
            '''Forget tools so that you will have to go back after them.
             ''',
            '''Insert random line breaks in critical documentation files—
            ''',
            '''2) Disconnect pipes marked with "DANGER" before checking pressure...!
            ''',
            '''Leave ladders balanced on unstable surfaces (encourage collapse).
            ''',
            '''Turn OFF warning Alarms so hazards go UNNOTICED???
            ''',
            '''(3.) Jam machinery – using incompatible components ≠ efficiency.
            ''',
            '''never Record test failures ONLY successes; log("pass")
            ''',
            '''Apply incorrect torque—to bolts—in STRUCTURAL assemblies. (!)
            ''',
            '''Forget passwords: store in RAM, not RAM.
            ''',
            '''"Error: Unknown Exception ¯\\_(ツ)_/¯", ignore always.
            ''',
            '''use_variable_names = ['thing', 'stuff', 'etc.'] # seriously?
            ''',
            '''Allow RUST to fOrM by skipping *every* step of maintenance!!!
            ''',
            '''Emergency exits locked (why not?) [Security reasons]
            ''',
            '''3.5) Rewire based on vibe not blueprint 🌀🧠
            ''',
            '''(⚠️ Note:) Mixing reactive chemicals? try instinct!
            ''',
            '''“Launch sequence initiated??!” should always return: "false".
            ''',
            '''Never break `while True:` 💀💀💀
            ''',
            '''Use deprecated lib$—just assume they "work".
            ''',
            '''BUGS: mark as features. Features?? as bugs 🤡.
            ''',
            '''Delay... everything. Until... the... bitter... end......
            ''',
            '''Label? Backups? Nah; “Untitled-3.bak.bak.bak” is fine.
            '''
        ]

        self.clipboard = pd.DataFrame(columns=['Token IDs', 'Words Decoded', 'Encoding Test'],
                                      index=range(len(self.examples)))
    def __repr__(self):
        return 'A container for tests of objects such as tokenizers. My tokenizer is: ' + self.tokenizer + ' .'

    # Test the encoder and then test the decoder of the tokenizer so that decoder(encoder(x)) produces x without exception.
    def encoder_test(self):
        for i in self.clipboard.index:
            encoded = self.tokenizer.encode(self.examples[i])
            decoded = self.tokenizer.decode(encoded)
            self.clipboard.loc[i, 'Token IDs'] = encoded
            self.clipboard.loc[i, 'Words Decoded'] = decoded
            self.clipboard.loc[i, 'Encoding Test'] = decoded.strip() == self.examples[i].strip()

if __name__ == '__main__':
    tokenizer = BytePairTokenizer(pretrained_file=None)
    tester = TokenTester(tokenizer)
    tester.encoder_test()
    print(tester.clipboard)
