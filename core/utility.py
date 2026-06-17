import torch

def text_to_token(text, tokenizer):
    encoded = tokenizer.encode(text)
    return (torch.tensor(encoded.ids).unsqueeze(0)) 

def token_to_text(token, tokenizer):
    decoded = tokenizer.decode((token.tolist()[0]))
    return (decoded)

def total_params(model, verbose=False):
    total_parameter = sum(p.numel() for p in model.parameters())
    if verbose:
        print("Total Parameter:", total_parameter)
    return total_parameter
