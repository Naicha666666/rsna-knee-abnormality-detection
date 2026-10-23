import torch
p = '/home/eason/rsna-knee/experiments/local_c14_64_k12_ep16_convnextv2_base_384_v6/corpus0_topk_avg.pt'
ck = torch.load(p, map_location='cpu', weights_only=False)
for k in ['arch', 'res', 'head', 'slots', 'gold_auc', 'weights']:
    print(k, ':', ck.get(k))
