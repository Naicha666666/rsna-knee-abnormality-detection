import sys
sys.path.insert(0, "/home/eason/rsna-knee/src")
import tv_stub
tv_stub.install()

import torch
import timm

print("Testing ConvNeXt-V2-Base forward on R9700...")
m = timm.create_model("convnextv2_base", pretrained=False, num_classes=0, in_chans=3, global_pool="").cuda()
sd = torch.load("/home/eason/rsna-knee/data/timm_weights/convnextv2_base_in22k_ft_in1k_384.pt", map_location="cpu")
sd = sd.get("model", sd)
m.load_state_dict(sd, strict=False)

x_zero = torch.zeros(2, 3, 336, 336, device="cuda")
out_zero = m(x_zero)
print("Zero input out nan:", torch.isnan(out_zero).any().item())

x_rand = torch.randn(2, 3, 336, 336, device="cuda")
out_rand = m(x_rand)
print("Rand input out nan:", torch.isnan(out_rand).any().item())
