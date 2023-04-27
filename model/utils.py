import torch.utils.data


def gather(c: torch.Tensor, t: torch.Tensor):
    return c.gather(-1, t).reshape(-1, 1, 1, 1)
