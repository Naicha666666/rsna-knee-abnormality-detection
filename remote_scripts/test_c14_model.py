import sys, os, torch
sys.path.insert(0, '/home/eason/rsna-knee/src')
sys.path.insert(0, '/home/eason/rsna-knee/pylib-cu')
import tv_stub; tv_stub.install()
import timm
from raptor import build_raptor_backbone, build_head

p = '/home/eason/rsna-knee/experiments/local_c14_64_k12_ep16_convnextv2_base_384_v6/corpus0_topk_avg.pt'
ck = torch.load(p, map_location='cpu', weights_only=False)
arch = ck.get('arch')
bb = build_raptor_backbone(arch)
model = build_head(bb, ck.get('head') or 'mil')
model.load_state_dict({k: v for k, v in ck['state_dict'].items()})
model.eval()
print('Model built and loaded strictly successfully!')
