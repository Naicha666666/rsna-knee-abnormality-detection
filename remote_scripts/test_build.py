import sys
sys.path.insert(0, "/home/eason/rsna-knee/src")
import raptor

arch = "convnextv2_base.fcmae_ft_in22k_in1k_384"
m = raptor.build_raptor_model(arch)
total_params = sum(p.numel() for p in m.parameters())
print(f"RAPTOR V2-Base built successfully! Param count: {total_params:,}")
