import tiktoken

def validate_model(model:str):
    try:
        enc = tiktoken.encoding_for_model(model)
        return enc
    except KeyError:
        raise ValueError(f"Unknown model: {model}")


def get_vocab_size(model:str | tiktoken.core.Encoding):
    if (isinstance(model, tiktoken.core.Encoding)):
        return model.n_vocab
    
    if (isinstance(model, str)):
        enc = validate_model(model)
        return (enc.n_vocab)
    
    raise TypeError("Model's type must be a string or tiktoken.core.Encoding")

def tokenize(enc: tiktoken.core.Encoding, text: str):
    encoded = enc.encode(text)
    return encoded


# with open("harrypotter.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()

# context_size = 6
# enc = tiktoken.get_encoding("gpt2")
# enc_text = tokenize(enc, raw_text)

# enc_sample = enc_text

# x = enc_sample[:context_size]
# y = enc_sample[1:context_size+1]

# print(f"x: {x}")
# print(f"y: {y}")
