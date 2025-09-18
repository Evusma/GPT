# parameters of the model - training
context_length = 16
model_dim = 12  # dimensionality for embedding and attention
num_blocks = 4  # number of repetitions of the transformer block
num_heads = 4  # number of self attention instances, each with size model_dim // num_heads

tokenizer = ByteTokenizer()

vocab_size = tokenizer.vocab_size
batch_size = 8
epochs = 10
lr=3e-4  # learning rate for the gradient descent method


# parameters of the model - resume training
context_length = 16
model_dim = 12  # dimensionality for embedding and attention
num_blocks = 4  # number of repetitions of the transformer block
num_heads = 4  # number of self attention instances, each with size model_dim // num_heads

tokenizer = ByteTokenizer()

vocab_size = tokenizer.vocab_size
batch_size = 8
epochs = 10
new_lr = 0.001  # new learning rate