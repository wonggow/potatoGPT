from pathlib import Path
from dataclasses import dataclass, field

# train.py

@dataclass
class GPTConfig:
    batch_size: int = 4 # Context_len * batch_size = input per process
    context_length: int = 128 # How many tokens should a tokenizer see.
    stride: int = 128 # How many tokens should a dataloader process before moving to next batch. Stride = Context_Length to prevent overlapping/underlapping
    shuffle: bool = True 

    dimension_out: int = 256 # Dimension Features, Vocab Size x Dimension
    head_number: int = 4 # DIMENSION OUT / HEAD_NUMBER = TOTAL DIM per HEAD
    head_layer: int = 4

    eval_freq: int = 10
    eval_iter: int = 1
    num_epochs: int = 5

    # Dataloader Ratio; train : eval
    train_ratio: float = 0.9 

    # Loss Function
    learning_rate: float = 4e-4
    weight_decay: float = 0.1

@dataclass
class DeviceConfig:
    device: str = "auto"
    start_context: list | str = "Harry Potter"

    # Model Training file
    model_training_files: list[str] | str = field(
        default_factory=lambda: [Path("dataset/harrypotter.txt")]
    )

    save_model_step: int = 10
    save_model_path: str = "model/checkpoint.pt"

@dataclass
class TokenizerConfig:
    tokenizer_training_files: list | str = field(
        default_factory= lambda: [Path("dataset/harrypotter.txt")]
    ) 

    vocab_size: int = 8000
    min_frequency: int =2

    add_bos_eos_processor: bool = False
    reserve_extra_tokens: int = 256 # Extra token backup

    tokenizer_save_path: str = "token_bpe.json"
