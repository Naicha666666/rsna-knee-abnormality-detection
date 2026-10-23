import sys, os, torch
sys.path.insert(0, '/home/eason/rsna-knee/src')
sys.path.insert(0, '/home/eason/rsna-knee/pylib-cu')
import tv_stub; tv_stub.install()
import timm

p = '/home/eason/rsna-knee/experiments/local_c14_64_k12_ep16_convnextv2_base_384_v6/corpus0_topk_avg.pt'
ck = torch.load(p, map_location='cpu', weights_only=False)
print('checkpoint loaded!')
print('arch:', ck.get('arch'))
print('res:', ck.get('res'))
print('gold_auc:', ck.get('gold_auc'))
print('keys:', list(ck.keys()))
