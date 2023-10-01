import torch

# Perform some operations that may leave unused memory in the cache
# ...

# Clear the cache
torch.cuda.empty_cache()
