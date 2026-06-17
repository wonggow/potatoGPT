from core.tokenizer import train_bytelevel_bpe_tokenizer
from config import TOKENIZER_TRAINING_FILES, VOCAB_SIZE, MIN_FREQUENCY, SAVE_PATH, ADD_BOS_EOS_PROCESSOR, RESERVE_EXTRA_TOKENS

train_bytelevel_bpe_tokenizer(
    files=TOKENIZER_TRAINING_FILES,
    vocab_size=VOCAB_SIZE,
    min_frequency=MIN_FREQUENCY,
    save_path=SAVE_PATH,
    add_bos_eos_processor= ADD_BOS_EOS_PROCESSOR,
    reserve_extra_tokens= RESERVE_EXTRA_TOKENS, 
    )