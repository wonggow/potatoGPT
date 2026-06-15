import torch
from tokenizers import Tokenizer

# train.py
BATCH_SIZE = 8 # Context_len * batch_size = input per process
CONTEXT_LENGTH = 64 # How many tokens should a tokenizer see.
STRIDE = 64 # How many tokens should a dataloader process before moving to next batch. Stride = Context_Length to prevent overlapping/underlapping
SHUFFLE = True 

DIMENSION_OUT = 512 # Dimension Features, Vocab Size x Dimension
HEAD_NUMBER = 4 # DIMENSION OUT / HEAD_NUMBER = TOTAL DIM per HEAD
HEAD_LAYER = 4

EVAL_FREQ = 5
EVAL_ITER = 5
START_CONTEXT = "Who are you"
NUM_EPOCHS = 10

# Loss Function
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 0.1

DEVICE = "auto"

# Training Tokenizer
FILES=["dataset/harrypotter.txt"] #list
VOCAB_SIZE = 8000
MIN_FREQUENCY=2
SAVE_PATH="token_bpe.json"
ADD_BOS_EOS_PROCESSOR= False
RESERVE_EXTRA_TOKENS= 256 # For fine-tuning extra token backup

TOKENIZER_PATH = "token_bpe.json"