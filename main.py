from llm_pretraining.dataloader import *
from llm_pretraining.tokenizer import *

import torch

with open("harrypotter.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader1 = create_dataloader(
    raw_text, batch_size=4, max_length=256, stride=256, shuffle=False, tokenizer=tiktoken.get_encoding("o200k_base")
)
dataloader2 = create_dataloader(
    raw_text, batch_size=4, max_length=256, stride=256, shuffle=False
)

# data_iter = iter(dataloader)
# first_batch = next(data_iter)
# second_batch = next(data_iter)
# print(first_batch)
# print(second_batch)

data_iter = iter(dataloader1)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)
data_iter = iter(dataloader2)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)