import os
import sys
sys.path.insert(0, "/home/eason/rsna-knee/src")
import tv_stub
tv_stub.install()
import timm
import torch

arch = "convnextv2_base.fcmae_ft_in22k_in1k_384"
print(f"Testing {arch} loading...")
try:
    m = timm.create_model(arch, pretrained=True, num_classes=0)
    print("SUCCESS: Loaded pretrained weights directly via timm/HF hub!")
    # Save offline weights to data/timm_weights/ for offline usage
    weights_path = "/home/eason/rsna-knee/data/timm_weights/convnextv2_base_in22k_ft_in1k_384.pt"
    if not os.path.exists(weights_path):
        torch.save(m.state_dict(), weights_path)
        print(f"Saved offline checkpoint to {weights_path}")
except Exception as e:
    print(f"Error loading pretrained model: {e}")
