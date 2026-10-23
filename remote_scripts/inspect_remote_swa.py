import torch
ckpt_path = '/home/eason/rsna-knee/data/weights/nartaa/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt'
ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
print('epoch:', ck.get('epoch'))
print('swa_over:', ck.get('swa_over'))
print('src:', ck.get('src'))
