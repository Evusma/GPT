This model is based on the GPT model explained by [Dev G](https://www.youtube.com/@gptLearningHub) in his video [Zero to Hero LLM Course](https://www.youtube.com/watch?v=F53Tt_vNLdg&t=6s)

I wanted to see for myself what training an LLM means, without a powerful laptop or compute 😅 

Later, after seeing that the loss didn’t improve, I realized that training isn’t that difficult. The key is the model parameters and their tuning 🤯

I started the training with the following model parameters:

context_length = 16
model_dim = 12  # dimensionality for embedding and attention
num_blocks = 4  # number of repetitions of the transformer block
num_heads = 4  # number of self-attention instances, each with size model_dim // num_heads
vocab_size = 258
batch_size = 8
epochs = 10
lr=3e-4  # learning rate for the gradient descent method

Later, I resumed the training with a learning rate of 0.001. In total, 40 epochs (around 4 hours)

The saved model isn’t very good yet, but I’ll keep improving it as long as the technical limitations allow 🙂 (I’m a bit scared of the training time for a model dimension of 64, as ChatGPT suggests 🤣)

In this repo, you’ll find:
- bon_jovi.txt -> The training data. It’s a copy of the Wikipedia page about the rock band *Bon Jovi*.
- byte_tokenizer -> A Python class with the tokenizer used for the model. It’s a basic tokenizer based on bytes.
- various .pth files -> Model checkpoints (weights only, architecture + weights, or architecture + weights + optimizer)
- class_gpt -> A Python class defining the model.
- class_testdataset -> A Python class to generate batch training data for the model.
- training.ipynb -> A Jupyter notebook to train the model.
- re-training.ipynb -> A Jupyter notebook to resume training from the last checkpoint.pth.
- evaluation.ipynb -> Coming soon.

