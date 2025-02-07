import torch

def enable_gpu_acceleration():
    """Configure and enable GPU acceleration for PyTorch using MPS (Metal Performance Shaders)"""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    if device.type == "mps":
        # Automatically configure for MPS
        torch.set_default_device(device)
        torch.set_float32_matmul_precision('high')
        
        # Optimized MPS configuration
        torch.backends.mps.set_cache_limit(0.5)  # Use 50% of GPU memory
        torch.backends.mps.set_allocator_settings("large")  # Optimize for large datasets
    
    return device

def verify_mps_support():
    """Verify MPS (Metal) support and print system information"""
    print(f"PyTorch {torch.__version__}")
    print("MPS available:", torch.backends.mps.is_available())
    print("MPS built:", torch.backends.mps.is_built())
    print(f"Using device: {torch.device('mps' if torch.backends.mps.is_available() else 'cpu')}") 