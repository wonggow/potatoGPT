from .tokenizer import *
import torch
from torch.utils.data import Dataset, DataLoader


class CustomGPTDataset(Dataset): # For format references -> https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
    def __init__(self, txt: str, tokenizer: tiktoken.core.Encoding, max_length: int, stride: int):
        self.input_ids = []                                 
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride): # Dont forget that python's range already implements -1 during the substraction.
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
                         tokenizer=None, num_workers=0):

    dataset = CustomGPTDataset(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader                                       



with open("harrypotter.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
# enc = tiktoken.get_encoding("gpt2")

# dataloader = create_dataloader(raw_text, tokenizer=enc, batch_size=8, max_length=4, stride=4, shuffle=False)

# data_iter = iter(dataloader)
# inputs, targets = next(data_iter)
# print("Inputs:\n", inputs)
# print("\nTargets:\n", targets)

# # gpt = CustomGPTDataset("Helo, my nawe is Steven Onggo. Im very pleased to meet you", enc, 5, 4)
# # create_dataloader(
# #     "Hello my name is Steven Onggo and I like transformers",
# #     batch_size=1,
# #     max_length=4,
# #     stride=4
# # )
# # print(enc.decode(gpt.input_ids))
# # print(gpt.target_ids)