from tokenizers import Tokenizer

samples = [ 
    "Hello world!",
    "Hello, who are you?",
    "Hello, what are you doing over here?",
    "你好， 你叫什么名字？",
    "   This is an indented code test\n\t with tabs!",
    "How about handling emoji? 😄🔥🚀",
    "Or even handling MATH?? ∑ x_i^2 = 42",
    "Or URLS?? https://example.com/ilove/buy?x=1",
]

tokenizer = Tokenizer.from_file("token_bpe.json")
for sample in samples:
    encoding = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoding.ids)

    print("Original: ", sample)
    print("Token IDs: ", encoding.ids)
    print("Tokens: ", encoding.tokens)
    print("Decoded: ", decoded)
    print("Equality: ", sample == decoded)
