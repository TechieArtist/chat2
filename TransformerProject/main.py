import torch
from data.download_data import download_data
from data.process_data import read_data, create_vocabulary, prepare_data, split_data, decode
from model.bigram_model import BigramLanguageModel
from train.train import train_model

# Parameters
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
file_name = 'data/input.txt'
block_size = 256
batch_size = 64
n_embd = 384
n_layer = 6
n_head = 6
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_iters = 27000
eval_interval = 500
eval_iters = 200
dropout = 0.2
patience = 150000

# Download and prepare data
download_data(url, file_name)
text = read_data(file_name)
chars, stoi, itos = create_vocabulary(text)
data = prepare_data(text, stoi)
train_data, val_data = split_data(data)

# Create model
model = BigramLanguageModel(vocab_size=len(chars), n_embd=n_embd, block_size=block_size, n_layer=n_layer, n_head=n_head).to(device)

# Train model
train_model(model, train_data, val_data, device, max_iters, eval_interval, eval_iters, patience, batch_size, block_size)

# Generate text after training
start_idx = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_indices = model.generate(start_idx, max_new_tokens=100, block_size=block_size)
generated_text = decode(generated_indices[0].tolist(), itos)
print(generated_text)

generated_indices = model.generate(start_idx, max_new_tokens=600, block_size=block_size)
generated_text = decode(generated_indices[0].tolist(), itos)
print(generated_text)
