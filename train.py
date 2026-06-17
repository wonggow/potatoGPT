from core.dataloader import create_dataloader
from core.tokenizer import load_tokenizer
from core.model import GPTModel, model_training
from core.utility import total_params
import torch.optim
import config as cfg

TOKENIZER = load_tokenizer(cfg.TOKENIZER_PATH)

DEVICE = torch.device(
    "cuda" if cfg.DEVICE == "auto" and torch.cuda.is_available()
    else "mps" if cfg.DEVICE == "auto" and torch.backends.mps.is_available()
    else "cpu" if cfg.DEVICE == "auto"
    else cfg.DEVICE
)  

texts = []

for file in cfg.MODEL_TRAINING_FILES:
    with open(file, "r", encoding="utf-8") as f:
        texts.append(f.read())

raw_text = "\n".join(texts)



split_data = int(cfg.TRAIN_RATIO * len(raw_text))
train_data = raw_text[:split_data]
eval_data = raw_text[split_data:]

train_dataloader = create_dataloader(
    train_data, batch_size=cfg.BATCH_SIZE, max_length=cfg.CONTEXT_LENGTH, stride=cfg.CONTEXT_LENGTH, shuffle=cfg.SHUFFLE, tokenizer=TOKENIZER
)

eval_dataloader = create_dataloader(
    eval_data, batch_size=cfg.BATCH_SIZE, max_length=cfg.CONTEXT_LENGTH, stride=cfg.CONTEXT_LENGTH, shuffle=False, tokenizer=TOKENIZER
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

total_params(model, verbose=True)

model_training(model=model, 
                train_loader=train_dataloader,
                eval_loader=eval_dataloader, 
                optimizer=OPTIMIZER, 
                device=DEVICE, 
                eval_freq= cfg.EVAL_FREQ, 
                eval_iter=cfg.EVAL_ITER, 
                start_context=cfg.START_CONTEXT, 
                num_epochs=cfg.NUM_EPOCHS,
                tokenizer=TOKENIZER)