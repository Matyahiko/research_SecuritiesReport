# setup_device.py

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup():
    """
    torchrun 専用
    """
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    print(f"Rank {dist.get_rank()} initialized with backend '{backend}'.")

def cleanup():
    dist.destroy_process_group()

def setup_device_optimizer_model_for_distributed(model, learning_rate, optimizer_class=torch.optim.AdamW):
    setup()
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
        torch.cuda.set_device(device)
        print(f"Rank {rank} is using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print(f"Rank {rank} is using CPU")
    
    model = model.to(device)
    ddp_model = DDP(model, device_ids=[device.index] if torch.cuda.is_available() else None)
    
    optimizer = optimizer_class(ddp_model.parameters(), lr=learning_rate)
    
    return device, optimizer, ddp_model
