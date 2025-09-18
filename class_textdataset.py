import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size):  #  here, block_size = context_length
        self.tokenizer = tokenizer
        self.block_size = block_size

        # Flatten all tokens into one big sequence
        all_tokens = []
        for t in texts:
            all_tokens.extend(tokenizer.encode(t))

        # chop the tokenized sequence into chunks of block_size
        self.data = []
        for i in range(0, max(1, len(all_tokens) - block_size)):
            x = all_tokens[i : i + block_size]
            y = all_tokens[i+1 : i + block_size+1]
            
            # pad if too short
            if len(x) < block_size:
                pad_len = block_size - len(x)
                x = x + [tokenizer.pad_token_id] * pad_len
                y = y + [tokenizer.pad_token_id] * pad_len
            
            self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)