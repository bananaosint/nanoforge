"""Replica-local losses: gather small sums/counts rather than full vocabulary logits."""
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F


class ReplicaLoss(nn.Module):
    def __init__(self,model,precision='float32',reference=None,beta=0.1):
        super().__init__()
        self.model,self.reference=model,reference
        self.precision,self.beta=precision,beta

    @staticmethod
    def log_probability(model,x,y):
        logits=model(x)['logits'].float().log_softmax(-1)
        return (logits.gather(-1,y.clamp_min(0).unsqueeze(-1)).squeeze(-1)*(y!=-100)).sum(-1)

    def forward(self,x,y,rejected_x=None,rejected_y=None):
        context=torch.autocast('cuda',dtype=torch.float16) if self.precision=='float16' else nullcontext()
        with context:
            if rejected_x is None:
                logits=self.model(x)['logits'].float()
                losses=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100,reduction='none')
                return losses.sum().reshape(1),(y!=-100).sum().reshape(1)
            policy=self.log_probability(self.model,x,y)-self.log_probability(self.model,rejected_x,rejected_y)
            with torch.no_grad():
                baseline=self.log_probability(self.reference,x,y)-self.log_probability(self.reference,rejected_x,rejected_y)
            losses=-F.logsigmoid(self.beta*(policy-baseline))
            return losses.sum().reshape(1),torch.ones_like(losses).sum().reshape(1)


def reduced_loss(result):
    total,count=result
    return total.sum()/count.sum().clamp_min(1)
