# AppleRGBDemoWithGPU (Windows 11 + RTX 3060)

## One‑command setup
Open **PowerShell** in this folder and run:
```powershell
.\install_gpu_env.ps1
```

This will:
1) Create **.venv_gpu**
2) Install **PyTorch (CUDA 12.1)** + vision/audio
3) Install extra deps from `requirements.txt`
4) Copy ready-to-train scripts into `scripts/`
5) Run a quick GPU test (`test_gpu.py`)
6) Save logs under `logs/`

## Train (after setup)
```powershell
. .\.venv_gpu\Scripts\Activate.ps1
# Create manifests
python scripts\make_manifests.py --root "H:\\Neuralbrion\\PlantVillage\\Solution\\ImageDataset\\img1" --outdir ".\manifests"

# Train & evaluate
python scripts\train_rgb.py --epochs 20 --amp --out runs\rgb_resnet18_gpu
python scripts\eval_rgb.py   --ckpt runs\rgb_resnet18_gpu\best_resnet18.pt --test_csv .\manifests\apple_test.csv --out runs\rgb_resnet18_gpu\eval_test

# Export & Grad-CAM
python scripts\export_onnx.py --ckpt runs\rgb_resnet18_gpu\best_resnet18.pt --out runs\rgb_resnet18_gpu\model.onnx --opset 14
python scripts\export_torchscript.py --ckpt runs\rgb_resnet18_gpu\best_resnet18.pt --out runs\rgb_resnet18_gpu\model.ts
python scripts\grad_cam.py --ckpt runs\rgb_resnet18_gpu\best_resnet18.pt --image "<path_to_image>" --out runs\rgb_resnet18_gpu\gradcam_heatmap.png --overlay runs\rgb_resnet18_gpu\gradcam_overlay.png
```
