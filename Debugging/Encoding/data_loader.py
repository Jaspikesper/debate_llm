import torch
from tokenizer import BytePairTokenizer
from torch.utils.data import DataLoader, Dataset
import model_config

class GPT_Dataset(Dataset): # Dataset of sequence/prediction pairs for a given text

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
        return torch.tensor(self.input_ids[idx], dtype=torch.long), torch.tensor(self.output_ids[idx],
                                                                                 dtype=torch.long)

def make_dataloader(dataset, shuffle=False, drop_last=False, num_workers=0, batch_size=None): # Inserts default parameters for Torch dataloader
    if batch_size is None:
        batch_size = model_config.GPT_CONFIG_124M["batch_size"]
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        batch_size=batch_size
    )
    return dataloader
