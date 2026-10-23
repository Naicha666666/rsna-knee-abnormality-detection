import torch

b = torch.load("/home/eason/rsna-knee/data/labels/purified_teacher_1200.pt")
print("=== Purified Teacher Bundle Metadata ===")
for k, v in b.get("metadata", {}).items():
    print(f"  {k}: {v}")
print(f"Total Unique UIDs: {len(b['uids'])}")
print("Registration Map:")
for p, m in b.get("registration_map", {}).items():
    print(f"  {p}: {m}")
