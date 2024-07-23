import torch

def read_data(file_name):
    """
    Reads data from a specified file.

    Args:
        file_name (str): The name of the file to read data from.

    Returns:
        str: The content of the file.
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def create_vocabulary(text):
    """
    Creates vocabulary mappings from the text.

    Args:
        text (str): The input text.

    Returns:
        tuple: A tuple containing characters, stoi, and itos mappings.
    """
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return chars, stoi, itos

def encode(text, stoi):
    """
    Encodes text into a list of integers.

    Args:
        text (str): The input text.
        stoi (dict): Character to index mapping.

    Returns:
        list: List of encoded integers.
    """
    return [stoi[c] for c in text]

def decode(encoded_text, itos):
    """
    Decodes a list of integers into text.

    Args:
        encoded_text (list): List of encoded integers.
        itos (dict): Index to character mapping.

    Returns:
        str: The decoded text.
    """
    return ''.join([itos[i] for i in encoded_text])

def prepare_data(text, stoi):
    """
    Prepares data by encoding text and converting to a tensor.

    Args:
        text (str): The input text.
        stoi (dict): Character to index mapping.

    Returns:
        torch.Tensor: Encoded text as a tensor.
    """
    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    return data

def split_data(data, split_ratio=0.9):
    """
    Splits data into training and validation sets.

    Args:
        data (torch.Tensor): The input data tensor.
        split_ratio (float): Ratio to split data into training and validation sets.

    Returns:
        tuple: Training and validation data tensors.
    """
    n = int(split_ratio * len(data))
    return data[:n], data[n:]
