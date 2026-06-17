from tokenizers import Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader


class CustomGPTDataset(Dataset):
    def __init__(self, txt: str, tokenizer: Tokenizer, max_length: int, stride: int):
        self.input_ids = []                                 
        self.target_ids = []

        encoding = tokenizer.encode(txt)
        token_ids = encoding.ids

        for i in range(0, len(token_ids) - max_length, stride): # -1
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
    
def create_dataloader(txt, batch_size=4, max_length=256, 
                         stride=256, shuffle=True, drop_last=True,
                         tokenizer=Tokenizer, num_workers=0):
    
    dataset = CustomGPTDataset(txt, tokenizer, max_length, stride)          

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

