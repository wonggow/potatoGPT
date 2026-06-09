import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, bias=False):
        super().__init__()
        assert d_model % n_heads == 0 # Split features must be same across heads

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = dropout      

    def forward(self, x):
        B, T, C = x.shape # Batch, Sequence Length, d_model || embedding_dimension

        qkv = self.qkv(x) # X * Wqkv
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) # split into n_heads, and each head receives head_dim features
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) # transpose to make it into [B, heads, T, head_dim]
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(                 
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)                 
    
class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  
        x = x + self.ffn(self.ln2(x))
        return x
    
class GPTModel(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, n_heads, n_layers, dropout=0.1):
        super().__init__()

        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.positional_embeddings = nn.Embedding(context_length, d_model)

        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape

        token_emb = self.token_embeddings(x)
        pos_emb = self.positional_embeddings(torch.arange(T, device=x.device))

        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)
        return logits       

def text_to_token(text, tokenizer):
    encoded = tokenizer.encode(text)
    return (torch.tensor(encoded.ids).unsqueeze(0)) 

def token_to_text(token, tokenizer):
    decoded = tokenizer.decode((token.tolist()[0]))
    return (decoded)

@torch.no_grad()
def generate_text(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens): # Autoregressive for each token in max_new_tokens.
        idx_cond = idx[:, -context_size:] # batch, context_size, vocab_size. take from -context_length to last token

        logits = model(idx_cond) # feed into model

        logits = logits[:, -1, :] # take last tensor of result

        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)
    return(idx)

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

def calc_loss_loader(data_loader, model, device, num_batches):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
            input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def generate_and_print_sample(model, tokenizer, device, text):
    model.eval()
    context_size = model.positional_embeddings.weight.shape[0]
    encoded = text_to_token(text, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_to_text(token_ids, tokenizer)
    print(decoded_text)
    model.train()
    
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
        train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
        val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

def model_training(model, train_loader, eval_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input, target in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input, target, model, device
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, eval_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                f"Train loss {train_loss:.3f}, "
                f"Val loss {val_loss:.3f}"
                )
            generate_and_print_sample(
                model, tokenizer, device, start_context="hello"
            )
    return train_losses, val_losses, track_tokens_seen