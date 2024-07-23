import torch

def get_batch(data, batch_size, block_size, device):
    """
    Generates a batch of data for training.

    Args:
        data (torch.Tensor): The dataset tensor.
        batch_size (int): Number of samples in a batch.
        block_size (int): Size of each input block.
        device (str): Device to load data on (cpu or cuda).

    Returns:
        tuple: A tuple containing input and target tensors.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters, batch_size, block_size, device):
    """
    Estimates the loss on training and validation data.

    Args:
        model (nn.Module): The model to evaluate.
        train_data (torch.Tensor): The training dataset.
        val_data (torch.Tensor): The validation dataset.
        eval_iters (int): Number of iterations for evaluation.
        batch_size (int): Number of samples in a batch.
        block_size (int): Size of each input block.
        device (str): Device to load data on (cpu or cuda).

    Returns:
        dict: Dictionary containing mean training and validation losses.
    """
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
