from core.dataloader import create_dataloader
from core.tokenizer import load_tokenizer
from core.model import GPTModel, model_training
from core.utility import total_params
import torch.optim
from pathlib import Path
from config import GPTConfig, TokenizerConfig, DeviceConfig

TOKENIZER = load_tokenizer(TokenizerConfig.tokenizer_save_path)

DEVICE = torch.device(
    "cuda" if DeviceConfig.device == "auto" and torch.cuda.is_available()
    else "mps" if DeviceConfig.device == "auto" and torch.backends.mps.is_available()
    else "cpu" if DeviceConfig.device == "auto"
    else DeviceConfig.device
)  

texts = []

for file in TokenizerConfig().tokenizer_training_files:
    with open(file, "r", encoding="utf-8") as f:
        texts.append(f.read())

raw_text = "\n".join(texts)



split_data = int(GPTConfig.train_ratio * len(raw_text))
train_data = raw_text[:split_data]
eval_data = raw_text[split_data:]

train_dataloader = create_dataloader(
    train_data, batch_size=GPTConfig.batch_size,
    max_length=GPTConfig.context_length,
    stride=GPTConfig.context_length,
    shuffle=GPTConfig.shuffle,
    tokenizer=TOKENIZER
)

eval_dataloader = create_dataloader(
    eval_data, 
    batch_size=GPTConfig.batch_size, 
    max_length=GPTConfig.context_length, 
    stride=GPTConfig.context_length, 
    shuffle=False, 
    tokenizer=TOKENIZER
)

model = GPTModel(
    vocab_size = TokenizerConfig.vocab_size,
    context_length = GPTConfig.context_length,
    d_model = GPTConfig.dimension_out,
    n_heads = GPTConfig.head_number,
    n_layers = GPTConfig.head_layer,
    )

file_path = Path(DeviceConfig.save_model_path)

if file_path.exists():
    model.load_state_dict(torch.load(file_path))

model = model.to(DEVICE)

OPTIMIZER = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)  

total_params(model, verbose=True)

model_training(model=model, 
                train_loader=train_dataloader,
                eval_loader=eval_dataloader, 
                optimizer=OPTIMIZER, 
                device=DEVICE, 
                eval_freq= GPTConfig.eval_freq, 
                eval_iter=GPTConfig.eval_iter, 
                start_context=DeviceConfig.start_context, 
                num_epochs=GPTConfig.num_epochs,
                save_step = DeviceConfig.save_model_step,
                save_file = DeviceConfig.save_model_path,
                tokenizer=TOKENIZER)