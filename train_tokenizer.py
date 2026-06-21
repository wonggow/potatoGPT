from core.tokenizer import train_bytelevel_bpe_tokenizer
from config import TokenizerConfig

cfg = TokenizerConfig()
train_bytelevel_bpe_tokenizer(
    files=cfg.tokenizer_training_files,
    vocab_size=cfg.vocab_size,
    min_frequency=cfg.min_frequency,
    save_path=cfg.tokenizer_save_path,
    add_bos_eos_processor= cfg.add_bos_eos_processor,
    reserve_extra_tokens= cfg.reserve_extra_tokens, 
    )