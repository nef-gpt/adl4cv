import torch


def decorator_timer(some_function):
    from time import time

    def wrapper(*args, **kwargs):
        t1 = time()
        result = some_function(*args, **kwargs)
        end = time() - t1
        return result, end

    return wrapper


def get_default_device():
    """
    Return either mps or cuda depending on the availability of the GPU
    Fall back to cpu if no GPU is available
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #    return torch.device("mps")
    else:
        return torch.device("cpu")
