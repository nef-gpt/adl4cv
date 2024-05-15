import torch


def get_default_device():
    """
    Return either mps or cuda depending on the availability of the GPU
    Fall back to cpu if no GPU is available
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
