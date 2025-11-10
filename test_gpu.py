import torch, time
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    x = torch.randn(1024, 1024, device='cuda')
    y = torch.randn(1024, 1024, device='cuda')
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(1000):
        z = x @ y
    torch.cuda.synchronize()
    print(f"Matmul 1000 iters on GPU took: {time.time()-t0:.3f}s")
else:
    print("Running CPU fallback benchmark...")
    x = torch.randn(512, 512)
    y = torch.randn(512, 512)
    t0 = time.time()
    for _ in range(200):
        z = x @ y
    print(f"Matmul 200 iters on CPU took: {time.time()-t0:.3f}s")
