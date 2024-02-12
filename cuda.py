
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the number of available CUDA devices
    num_devices = torch.cuda.device_count()

    # Print information about each CUDA device
    for i in range(num_devices):
        device = torch.cuda.get_device_name(i)
        print(f"CUDA Device {i}: {device}")
else:
    print("CUDA is not available on this system.")
    
cuda.py
Ipinapakita ang cuda.py.
