import torch

print("CUDA Available:", torch.cuda.is_available())              # Should be True
print("Device Name:", torch.cuda.get_device_name(0))             # Should show RTX 4060
print("Torch Version:", torch.__version__)
