import torch
from tokenizers import Tokenizer

# train.py
BATCH_SIZE = 4 # Context_len * batch_size = input per process
CONTEXT_LENGTH = 256 # How many tokens should a tokenizer see.
STRIDE = 256 # How many tokens should a dataloader process before moving to next batch. Stride = Context_Length to prevent overlapping/underlapping
SHUFFLE = True 

DIMENSION_OUT = 1024 # Dimension Features, Vocab Size x Dimension
HEAD_NUMBER = 4 # DIMENSION OUT / HEAD_NUMBER = TOTAL DIM per HEAD
HEAD_LAYER = 4

EVAL_FREQ = 5
EVAL_ITER = 5
START_CONTEXT = "Harry Potter"
NUM_EPOCHS = 10

# Dataloader Ratio; train : eval
TRAIN_RATIO = 0.9 

# Loss Function
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 0.1

DEVICE = "auto"

# Model Training file
MODEL_TRAINING_FILES = ["dataset/harrypotter.txt"]

# Training Tokenizer
TOKENIZER_TRAINING_FILES = ["dataset/harrypotter.txt"] #list
VOCAB_SIZE = 8000
MIN_FREQUENCY=2
SAVE_PATH="token_bpe.json"
ADD_BOS_EOS_PROCESSOR= False
RESERVE_EXTRA_TOKENS= 256 # For fine-tuning extra token backup

TOKENIZER_PATH = "token_bpe.json"