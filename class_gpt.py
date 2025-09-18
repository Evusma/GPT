import torch
import torch.nn as nn
import torch.nn.functional as F

class GPT(nn.Module):
    
    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.token_embeddings = nn.Embedding(vocab_size, model_dim)
        self.pos_embeddings = nn.Embedding(context_length, model_dim)
        # number of iterations trough the transformer block
        self.blocks = nn.Sequential()
        for i in range(num_blocks):
            self.blocks.append(self.TransformerBlock(model_dim, num_heads))
        self.final_ln = nn.LayerNorm(model_dim)
        self.vocabulary_projection = nn.Linear(model_dim, vocab_size)
                   
    def forward(self, context):
        # context: TensorType[int] -> TensorType[float]
        torch.manual_seed(0)
        token_embeds = self.token_embeddings(context)  # B, T, D
        B, T, D = token_embeds.shape
        pos_embeds = self.pos_embeddings(torch.arange(T))
        total_embeddings = token_embeds + pos_embeds
        
        un_normalized = self.vocabulary_projection(self.final_ln(self.blocks(total_embeddings)))
        probs = nn.functional.softmax(un_normalized, dim = -1)
        return probs
        
    class TransformerBlock(nn.Module):
        
        class MultiHeadedSelfAttention(nn.Module):
        
            class SingleHeadAttention(nn.Module):
                
                def __init__(self, model_dim, head_size):
                    super().__init__()
                    torch.manual_seed(0)
                    # not biases in the linear layers for getting the keys, queries and values of the tokens (for attention, better results)
                    self.get_keys = nn.Linear(model_dim, head_size, bias=False)
                    self.get_queries = nn.Linear(model_dim, head_size, bias=False)
                    self.geet_values = nn.Linear(model_dim, head_size, bias=False)

                def forward(self, embedded):
                    k = self.get_keys(embedded)  # BxTxA
                    q = self.get_queries(embedded)
                    v = self.geet_values(embedded)

                    scores = q @ torch.transpose(k, 1, 2)
                    b, t, a = k.shape  # batch dim, context dim, attention dim
                    scores = scores/ (a ** 0.5)

                    # lower triangular tensor
                    pre_mask = torch.tril(torch.ones(t, t))
                    mask = pre_mask == 0

                    scores = scores.masked_fill(mask, float('-inf'))  # b, t, t
                    scores = nn.functional.softmax(scores, dim=2)  # dim=2 is the columns
                    transformed = scores @ v

                    return transformed
        
            def __init__(self, model_dim, num_heads):
                super().__init__()
                torch.manual_seed(0)
                self.heads = nn.ModuleList()  # list to store neural network layers
                for i in range(num_heads):
                    # list of single attention layers
                    self.heads.append(self.SingleHeadAttention(model_dim, model_dim // num_heads))

            def forward(self, embedded):
                outputs = []  # each element is B, T, Head_size --> B, T, Attention_sim (after concatenation)
                for head in self.heads:
                    outputs.append(head(embedded))
                cated = torch.cat(outputs, dim = 2)  # dim = 2 the last dimension (attention)
                return cated
    
        def __init__(self, model_dim, num_heads):
            super().__init__()
            torch.manual_seed(0)
            # multi head self attention layer
            self.mhsa = self.MultiHeadedSelfAttention(model_dim, num_heads)
            # layers norm
            self.first_ln = nn.LayerNorm(model_dim)
            self.second_ln = nn.LayerNorm(model_dim)
            # fee forward
            self.ff = self.VanillaNeuralNetwork(model_dim)
        

        def forward(self, embedded):
            # def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            torch.manual_seed(0)
            # add layer after the multi head self attention
            first_part = embedded + self.mhsa(self.first_ln(embedded))
            # add layer after feed forward
            result = first_part + self.ff(self.second_ln(first_part))
            return result
    
        class VanillaNeuralNetwork(nn.Module):
        
            def __init__(self, model_dim, droput=0.1):
                super().__init__()
                self.fc1 = nn.Linear(model_dim, model_dim)
                self.fc2 = nn.Linear(model_dim, model_dim)
                self.dropout = nn.Dropout()

            def forward(self, x):
                x = self.fc1(x)
                x = F.relu(x)
                x = self.dropout(x)
                x = self.fc2(x)
                x = self.dropout(x)
                return x
    
    

