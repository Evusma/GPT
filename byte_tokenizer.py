class ByteTokenizer:
    """
    UTF-8 byte tokenizer: every byte (0–255) is a token.
    Reserve extra IDs for special tokens (eos, pad).
    """
    def __init__(self):
        self.vocab_size = 258
        self.eos_token_id = 256
        self.pad_token_id = 257

    def encode(self, text: str):
        b = text.encode("utf-8", errors="ignore")
        return list(b) + [self.eos_token_id]

    def decode(self, ids):
        b = bytes([i for i in ids if i < 256])
        return b.decode("utf-8", errors="replace")


