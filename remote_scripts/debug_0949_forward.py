import torch
import torch.nn as nn
import timm
import sys
sys.path.insert(0, '/home/eason/rsna-knee/src')
import tv_stub
tv_stub.install()

ck = torch.load('/home/eason/rsna-knee/data/weights/nartaa/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt', map_location='cpu', weights_only=False)

class NartaaRaptor(nn.Module):
    def __init__(self, bb, f_dim=768, n=12):
        super().__init__()
        self.backbone = bb
        self.norm = nn.LayerNorm(f_dim)
        self.att = nn.Sequential(nn.Linear(f_dim, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, f_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
    def encode(self, x):
        B, K = x.shape[:2]
        flat = x.flatten(0, 1)
        f = torch.cat([self.backbone(chunk) for chunk in flat.split(16)], dim=0)
        return f.view(B, K, -1)
    def head(self, feats):
        h = self.norm(feats)
        a = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.clsW).sum(-1) + self.clsb
    def forward(self, x):
        return self.head(self.encode(x))

bb = timm.create_model('coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', pretrained=False, num_classes=0, in_chans=3, img_size=320, global_pool='avg')
m = NartaaRaptor(bb, f_dim=bb.num_features).cuda()
m.load_state_dict(ck['model'])
m.eval()

# Test dummy forward pass
x = torch.randn(1, 42, 3, 320, 320).cuda()
with torch.no_grad():
    logits = m(x)
    probs = torch.sigmoid(logits)

print("Dummy logits shape:", logits.shape)
print("Dummy logits:", logits[0].cpu().numpy())
print("Dummy probs:", probs[0].cpu().numpy())
