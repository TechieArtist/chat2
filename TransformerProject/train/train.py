import torch
from train.utils import get_batch, estimate_loss
from model.bigram_model import BigramLanguageModel

def train_model(model, train_data, val_data, device, max_iters, eval_interval, eval_iters, patience, batch_size, block_size):
    """
    Trains the model with early stopping.

    Args:
        model (nn.Module): The model to train.
        train_data (torch.Tensor): The training dataset.
        val_data (torch.Tensor): The validation dataset.
        device (str): Device to load data on (cpu or cuda).
        max_iters (int): Maximum number of training iterations.
        eval_interval (int): Interval for evaluating the model.
        eval_iters (int): Number of iterations for evaluation.
        patience (int): Patience for early stopping.
        batch_size (int): Number of samples in a batch.
        block_size (int): Size of each input block.

    Returns:
        None
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val_loss = float('inf')
    wait = 0

    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, eval_iters, batch_size, block_size, device)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                wait = 0
            else:
                wait += eval_interval
                if wait >= patience:
                    print("Early stopping!")
                    break

        xb, yb = get_batch(train_data, batch_size, block_size, device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
