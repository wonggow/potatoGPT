from pathlib import Path
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFC
from tokenizers.processors import TemplateProcessing


BASE_SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    "<|end_of_turn|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
]

REQUIRED_TOKENS = [
    "<|pad|>", 
    "<|bos|>", 
    "<|eos|>", 
    "<|unk|>"]

def train_bytelevel_bpe_tokenizer(
    files: list[str],
    vocab_size: int = 128000,
    min_frequency: int = 2,
    save_path: str = "token_bpe.json",
    add_bos_eos_processor: bool = False,
    reserve_extra_tokens: int = 256, ) -> Tokenizer:

    file_paths = [] 
    for file in files: #file_paths sanity check
        path = Path(file)
        
        if not path.exists():
            raise FileNotFoundError(f"Training file does not exists: {file}")
        if not path.is_file():
            raise ValueError(f"Training path does not exists: {file}")
        else:
            file_paths.append(str(path))

    special_tokens = BASE_SPECIAL_TOKENS + [
        f"<|reserved_{i}|>" for i in range(reserve_extra_tokens)
    ] # Add base tokens + reserved tokens for future upcoming tokens if required

    if vocab_size <= (len(special_tokens) + len(ByteLevel.alphabet())):
        raise ValueError(
            f"{vocab_size} is too small. "
            f"Minimum vocab_size is {len(special_tokens) + len(ByteLevel.alphabet()) } for special tokens + byte alphabet."
        )
    
    tokenizer = Tokenizer(
        BPE(
            unk_token="<|unk|>",
            byte_fallback=False,  # ByteLevel already covers raw bytes.
        )
    )
    
    tokenizer.normalizer = NFC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True, )
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[ 
            AddedToken(token, normalized=False)
            for token in special_tokens
        ],
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    tokenizer.train(file_paths, trainer)

    for token in REQUIRED_TOKENS:
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            raise RuntimeError(f"Special token missing after training: {token}")
        
    if add_bos_eos_processor:
        tokenizer.post_processor = TemplateProcessing(
            single="<|bos|> $A <|eos|>",
            pair="<|bos|> $A <|eos|> $B:1 <|eos|>:1",
            special_tokens=[
                ("<|bos|>", tokenizer.token_to_id("<|bos|>")),
                ("<|eos|>", tokenizer.token_to_id("<|eos|>")),
            ],
        )

    tokenizer.save(save_path)
    return tokenizer

def load_tokenizer(path: str = "token_bpe.json") -> Tokenizer:
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {path}")

    return Tokenizer.from_file(str(path_obj))
