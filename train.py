from core.dataloader import *
from core.tokenizer import *
from core.model import *
from datasets import load_dataset
import torch.optim

import config as cfg

TOKENIZER = Tokenizer.from_file(cfg.TOKENIZER_PATH)

DEVICE = torch.device(
    "cuda" if cfg.DEVICE == "auto" and torch.cuda.is_available()
    else "mps" if cfg.DEVICE == "auto" and torch.backends.mps.is_available()
    else "cpu" if cfg.DEVICE == "auto"
    else cfg.DEVICE
)  

with open("dataset/harrypotter.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()


# with open("dataset/train.txt", "r", encoding="utf-8") as f:
#     raw2_text = f.read()


# raw_text = raw_text + raw2_text
dataloader = create_dataloader(
    raw_text, batch_size=cfg.BATCH_SIZE, max_length=cfg.CONTEXT_LENGTH, stride=cfg.CONTEXT_LENGTH, shuffle=cfg.SHUFFLE, tokenizer=TOKENIZER
)

model = GPTModel(
    vocab_size = cfg.VOCAB_SIZE,
    context_length = cfg.CONTEXT_LENGTH,
    d_model = cfg.DIMENSION_OUT,
    n_heads = cfg.HEAD_NUMBER,
    n_layers = cfg.HEAD_LAYER,
    )

model = model.to(DEVICE)

OPTIMIZER = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)  

model_training(model=model, 
                train_loader=dataloader,
                eval_loader=dataloader, 
                optimizer=OPTIMIZER, 
                device=DEVICE, 
                eval_freq= cfg.EVAL_FREQ, 
                eval_iter=cfg.EVAL_ITER, 
                start_context=cfg.START_CONTEXT, 
                num_epochs=cfg.NUM_EPOCHS,
                tokenizer=TOKENIZER)