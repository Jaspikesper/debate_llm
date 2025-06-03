from urllib.request import urlretrieve
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import tempfile
import os
import sys

# Add the parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from tokenizer import BytePairTokenizer
except ImportError:
    # Try alternative import path
    sys.path.append('C:/Users/jaspe/PycharmProjects/PythonProject7/Encoding')
    from tokenizer import BytePairTokenizer


# Import the dataloader components
class GPT_Dataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride=1):
        self.ids = tokenizer.encode(text)
        self.input_ids = []
        self.output_ids = []
        i = 0
        while i + max_length < len(self.ids):
            input_chunk = self.ids[i:i + max_length]
            output_chunk = self.ids[i + 1:i + 1 + max_length]
            self.input_ids.append(input_chunk)
            self.output_ids.append(output_chunk)
            i += stride

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx], dtype=torch.long), torch.tensor(self.output_ids[idx], dtype=torch.long)


def make_dataloader(dataset, shuffle=False, drop_last=False, num_workers=0, batch_size=6):
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        batch_size=batch_size
    )
    return dataloader


rawtext_link = 'https://www.gutenberg.org/ebooks/26184.txt.utf-8'
urlretrieve(rawtext_link, '../byte_training/LLM_dataset.txt')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)


class DataloaderTester:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.examples = [
            '''Machine learning requires proper data preprocessing for optimal results.
            ''',
            '''The GPT dataset creates input-output sequence pairs automatically.
            ''',
            '''Tokenization converts raw text into numerical token representations.
            ''',
            '''DataLoader handles batching and shuffling of training examples efficiently.
            ''',
            '''Stride parameters control overlapping windows in sequence generation tasks.
            ''',
            '''Unit tests verify correctness of data pipeline implementation components.
            ''',
            '''Batch processing improves computational efficiency during model training phases.
            ''',
            '''Text encoding must preserve information while enabling neural network processing.
            ''',
            '''PyTorch tensors facilitate automatic differentiation and GPU acceleration.
            ''',
            '''Language models learn patterns from sequential text data representations.
            ''',
            '''Dataset classes define how individual samples are accessed and processed.
            ''',
            '''Proper validation ensures robustness of machine learning data pipelines.
            ''',
            '''Embedding layers map discrete tokens to continuous vector representations.
            ''',
            '''Training loops iterate through batches to optimize model parameters.
            ''',
            '''Sequence modeling captures dependencies between adjacent text elements.
            ''',
            '''Data augmentation techniques can improve model generalization capabilities significantly.
            ''',
            '''Cross-validation helps assess model performance on unseen data samples.
            ''',
            '''Hyperparameter tuning optimizes learning rates and architectural choices.
            ''',
            '''Gradient descent algorithms minimize loss functions through iterative updates.
            ''',
            '''Neural networks approximate complex functions through learned transformations.
            ''',
            '''Attention mechanisms allow models to focus on relevant input portions.
            ''',
            '''Transformer architectures have revolutionized natural language processing tasks.
            ''',
            '''Fine-tuning adapts pre-trained models to specific downstream applications.
            ''',
            '''Regularization techniques prevent overfitting on limited training datasets.
            ''',
            '''Model evaluation requires comprehensive metrics beyond simple accuracy scores.
            '''
        ]
        self.clipboard = pd.DataFrame(columns=['Max_Length', 'Stride', 'Batch_Size', 'Dataset_Length',
                                               'Batch_Count', 'Input_Shape', 'Target_Shape',
                                               'Sequence_Shift_Test', 'Decode_Test'],
                                      index=range(len(self.examples)))

    def __repr__(self):
        return 'A container for tests of GPT_Dataset and DataLoader functionality. My tokenizer is: ' + str(
            type(self.tokenizer).__name__) + ' .'

    def dataloader_test(self):
        test_configs = [
            {'max_length': 3, 'stride': 1, 'batch_size': 2},
            {'max_length': 5, 'stride': 2, 'batch_size': 3},
            {'max_length': 4, 'stride': 1, 'batch_size': 1},
            {'max_length': 6, 'stride': 3, 'batch_size': 4},
            {'max_length': 2, 'stride': 1, 'batch_size': 5}
        ]

        for i in self.clipboard.index:
            if i < len(test_configs):
                config = test_configs[i]
            else:
                # Use cycling configs for remaining examples
                config = test_configs[i % len(test_configs)]

            try:
                # Create dataset
                dataset = GPT_Dataset(
                    text=self.examples[i],
                    tokenizer=self.tokenizer,
                    max_length=config['max_length'],
                    stride=config['stride']
                )

                # Create dataloader
                dataloader = make_dataloader(
                    dataset,
                    batch_size=config['batch_size'],
                    shuffle=False
                )

                # Test first batch
                if len(dataset) > 0:
                    inputs, targets = next(iter(dataloader))

                    # Test sequence shift relationship
                    sequence_shift_correct = True
                    if inputs.shape[0] > 0:
                        for batch_idx in range(inputs.shape[0]):
                            input_seq = inputs[batch_idx].tolist()
                            target_seq = targets[batch_idx].tolist()
                            # Target should be input shifted by 1 position
                            if input_seq[1:] != target_seq[:-1]:
                                sequence_shift_correct = False
                                break

                    # Test decoding
                    decode_success = True
                    try:
                        decoded_input = self.tokenizer.decode(inputs[0].tolist())
                        decoded_target = self.tokenizer.decode(targets[0].tolist())
                    except Exception as decode_error:
                        decode_success = False
                        print(f"Decode error for example {i}: {decode_error}")

                    # Count total batches
                    batch_count = sum(1 for _ in dataloader)

                    # Record results
                    self.clipboard.loc[i, 'Max_Length'] = config['max_length']
                    self.clipboard.loc[i, 'Stride'] = config['stride']
                    self.clipboard.loc[i, 'Batch_Size'] = config['batch_size']
                    self.clipboard.loc[i, 'Dataset_Length'] = len(dataset)
                    self.clipboard.loc[i, 'Batch_Count'] = batch_count
                    self.clipboard.loc[i, 'Input_Shape'] = str(tuple(inputs.shape))
                    self.clipboard.loc[i, 'Target_Shape'] = str(tuple(targets.shape))
                    self.clipboard.loc[i, 'Sequence_Shift_Test'] = sequence_shift_correct
                    self.clipboard.loc[i, 'Decode_Test'] = decode_success

                else:
                    # Empty dataset case
                    self.clipboard.loc[i, 'Max_Length'] = config['max_length']
                    self.clipboard.loc[i, 'Stride'] = config['stride']
                    self.clipboard.loc[i, 'Batch_Size'] = config['batch_size']
                    self.clipboard.loc[i, 'Dataset_Length'] = 0
                    self.clipboard.loc[i, 'Batch_Count'] = 0
                    self.clipboard.loc[i, 'Input_Shape'] = 'N/A'
                    self.clipboard.loc[i, 'Target_Shape'] = 'N/A'
                    self.clipboard.loc[i, 'Sequence_Shift_Test'] = False
                    self.clipboard.loc[i, 'Decode_Test'] = False

            except Exception as e:
                print(f"Error processing example {i}: {e}")
                # Error case
                self.clipboard.loc[i, 'Max_Length'] = config['max_length']
                self.clipboard.loc[i, 'Stride'] = config['stride']
                self.clipboard.loc[i, 'Batch_Size'] = config['batch_size']
                self.clipboard.loc[i, 'Dataset_Length'] = 'ERROR'
                self.clipboard.loc[i, 'Batch_Count'] = 'ERROR'
                self.clipboard.loc[i, 'Input_Shape'] = 'ERROR'
                self.clipboard.loc[i, 'Target_Shape'] = 'ERROR'
                self.clipboard.loc[i, 'Sequence_Shift_Test'] = False
                self.clipboard.loc[i, 'Decode_Test'] = False


if __name__ == '__main__':
    try:
        # Try to initialize tokenizer with vocab file
        vocab_path = '../byte_training/tok.json'
        if os.path.exists(vocab_path):
            tokenizer = BytePairTokenizer(vocab_file=vocab_path)
        else:
            # Try absolute path
            vocab_path = 'C:/Users/jaspe/PycharmProjects/PythonProject7/Encoding/byte_training/tok.json'
            if os.path.exists(vocab_path):
                tokenizer = BytePairTokenizer(vocab_file=vocab_path)
            else:
                print("Warning: vocab file not found, using None")
                tokenizer = BytePairTokenizer(vocab_file=None)

        print(f"Tokenizer initialized successfully!")
        print(f"Vocab size: {len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else 'Unknown'}")

        tester = DataloaderTester(tokenizer)
        tester.dataloader_test()
        print(tester.clipboard)

    except Exception as e:
        print(f"Error initializing test: {e}")
        import traceback

        traceback.print_exc()