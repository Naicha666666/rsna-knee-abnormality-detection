import torch

print("torch version:", torch.__version__)
print("hip version:", getattr(torch.version, 'hip', None))
print("cuda is_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Total memory: {props.total_memory / (1024**3):.2f} GB")
        print(f"  Capability: {props.major}.{props.minor}")
    
    # Test tensor allocation
    x = torch.randn(1000, 1000, device='cuda')
    y = x @ x
    print("Tensor MatMul OK on CUDA device! Norm:", y.norm().item())
else:
    print("CUDA is NOT available in this environment!")
