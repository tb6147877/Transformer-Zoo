
import torch
def get_device(verbose: bool = False) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print("Using GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print("Using Apple Silicon GPU")
    else:
        device = torch.device("cpu")
        if verbose:
            print("Using CPU")
    return device