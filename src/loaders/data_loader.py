import math
import torch
import numpy as np

# def data_loader(dataset,  batch_size: int, context_length: int, device: str): 
#     indices = np.arange(len(dataset) - context_length)
#     sampled_indices = np.random.choice(indices, size=batch_size, replace=False)
#     dataset = np.stack([dataset for _ in range(batch_size)], axis=0)
#     indices = [np.arange(sampled_index, sampled_index + context_length + 1) for sampled_index in sampled_indices]
#     dataset_indices = np.stack([dataset[i, indices[i]] for i in range(batch_size)], axis=0)
#     input_ids = torch.from_numpy(dataset_indices[:,:-1]).to(device)
#     labels = torch.from_numpy(dataset_indices[:,1:]).to(device)
#     return input_ids, labels

def make_position_ids(input_ids: torch.Tensor, eos_token_id: int):
    """
    input_ids: (B, S) LongTensor
    Reset rule: positions start at 0; after every EOS token, the *next* token restarts at 0.
    """
    B, S = input_ids.shape
    device = input_ids.device

    # segment starts: col 0, and any position whose *previous* token was EOS
    starts = torch.zeros_like(input_ids, dtype=torch.bool)
    starts[:, 0] = True
    starts[:, 1:] = (input_ids[:, :-1] == eos_token_id)

    # index grid
    idx = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # index of the most recent segment start, per position (running max trick)
    last_start_idx = torch.where(starts, idx, torch.full_like(idx, -1))
    last_start_idx = torch.cummax(last_start_idx, dim=1).values  # (B, S)

    position_ids = (idx - last_start_idx).to(torch.long)

    return position_ids

def data_loader(dataset, batch_size: int, context_length: int, eos_token_id: int, device: str = 'cuda'): 

    range = len(dataset) - context_length
    rng = np.random.default_rng()          # fastest generator in NumPy ≥1.17
    sampled = rng.choice(range, batch_size, replace=False)
    batch_indices = sampled[:, None] + np.arange(context_length + 1)[None, :] # advanced indexing
    batch_data = dataset[batch_indices]
    input_ids = batch_data[:, :-1]
    labels = batch_data[:, 1:]
        
    input_ids = torch.tensor(input_ids, dtype=torch.int32).to(device)
    labels = torch.tensor(labels, dtype=torch.int64).to(device)
    position_ids = make_position_ids(input_ids, eos_token_id)
    position_ids = torch.tensor(position_ids, dtype=torch.int64).to(device)

    return input_ids, labels, position_ids