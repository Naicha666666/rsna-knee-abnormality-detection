import torch

p = "/home/eason/rsna-knee/experiments/s6_convnextv2_base_fold0/s6_convnextv2_base_fold0_best.pt"
ckpt = torch.load(p, map_location="cpu")
print("=== S6 Best Checkpoint Info ===")
print("Keys:", list(ckpt.keys()))
print("Epoch:", ckpt.get("epoch"))
print("Fold:", ckpt.get("fold"))
print("Gold AUC:", ckpt.get("gold_auc"))
print("Train Loss:", ckpt.get("train_loss"))
print("Backbone:", ckpt.get("backbone"))
sd = ckpt["model_state"]
prefixes = set(k.split(".")[0] for k in sd.keys())
print("Prefixes:", sorted(list(prefixes)))
print("Total tensor count:", len(sd))
print("sp_q shape:", sd["sp_q"].shape)
print("win_attn_q shape:", sd["win_attn_q"].shape)
print("head weight shape:", sd["head.weight"].shape)
