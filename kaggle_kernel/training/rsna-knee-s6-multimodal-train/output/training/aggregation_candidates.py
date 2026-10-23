"""Small, explicit consumers for score summaries and cross-Series relations."""
import math
import torch
from torch import nn


def score_summary(logits):
    """N windows x L labels -> L x 8; one real Series/view at a time."""
    if logits.ndim!=2 or not len(logits):raise ValueError('nonempty window logits required')
    p=logits.float().sigmoid()
    top=p.topk(max(1,math.ceil(len(p)*.1)),dim=0).values.mean(0)
    quantiles=torch.quantile(p,torch.tensor([.5,.9,.99],device=p.device),dim=0)
    return torch.stack([p.mean(0),p.amax(0),top,*quantiles,p.std(0,unbiased=False),(p>.5).float().mean(0)],-1)


class ScoreSummary(nn.Module):
    def __init__(self,hidden=32):
        super().__init__();self.net=nn.Sequential(nn.Linear(8,hidden),nn.GELU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,logits):return self.net(score_summary(logits)).squeeze(-1)


class PairwiseSeries(nn.Module):
    """Bounded feature relations; no fictitious pixel/slice registration."""
    def __init__(self,dim,hidden,max_partners=2):
        super().__init__();self.max_partners=max_partners
        self.net=nn.Sequential(nn.Linear(dim*3,hidden),nn.GELU(),nn.Linear(hidden,dim))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,values,slots,original_series):
        outputs=[]
        for i in range(len(values)):
            # Exclude duplicate views of the same acquisition and identical
            # protocol types. Preserve every Series in the main aggregation.
            partners=[j for j in range(len(values)) if original_series[j]!=original_series[i] and slots[j]!=slots[i]]
            partners=partners[:self.max_partners]
            if not partners:outputs.append(torch.zeros_like(values[i]));continue
            left=values[i].expand(len(partners),-1,-1);right=values[partners]
            outputs.append(self.net(torch.cat([left,right,left-right],-1)).mean(0))
        return values+torch.stack(outputs)
