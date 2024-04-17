from yacs.config import CfgNode
import torch


def update_config(d, u):
    for k, v in u.items():
        if isinstance(v, CfgNode):
            d[k] = update_config(d.get(k, CfgNode()), v)
        else:
            d[k] = v
    return d


def list2batch(tensors):
    num_tensors = len(tensors)
    max_len = max([len(t) for t in tensors])
    last_dim = tensors[0].shape[-1]
    dtype_, device_ = tensors[0].dtype, tensors[0].device
    batched_tensor = torch.zeros((num_tensors, max_len, last_dim), dtype=dtype_, device=device_)
    mask = batched_tensor[..., 0].bool()
    for idx, t in enumerate(tensors):
        batched_tensor[idx][:len(t)] = t
        mask[idx][:len(t)] = True
    
    return batched_tensor, mask
