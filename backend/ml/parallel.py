"""Replica-local losses: gather small sums/counts rather than full vocabulary logits."""
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F


class ReplicaLoss(nn.Module):
    def __init__(self,model,precision='float32',reference=None,beta=0.1,telemetry=False):
        super().__init__()
        self.model,self.reference=model,reference
        self.precision,self.beta=precision,beta
        self.telemetry=telemetry

    @staticmethod
    def log_probability(model,x,y):
        output=model(x)
        logits=output['logits'].float().log_softmax(-1)
        return (logits.gather(-1,y.clamp_min(0).unsqueeze(-1)).squeeze(-1)*(y!=-100)).sum(-1),output['aux_loss']

    def forward(self,x,y,rejected_x=None,rejected_y=None):
        context=torch.autocast('cuda',dtype=torch.bfloat16 if self.precision=='bfloat16' else torch.float16) if self.precision in ('float16','bfloat16') else nullcontext()
        with context:
            if rejected_x is None:
                output=self.model(x)
                logits=output['logits'].float()
                losses=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100,reduction='none')
                count=(y!=-100).sum()
                result=((losses.sum()+output['aux_loss']*count).reshape(1),count.reshape(1))
                if self.telemetry:
                    weight=x.numel()
                    return (*result,(output['balance_loss']*weight).reshape(1),(output['z_loss']*weight).reshape(1),
                            (output['expert_load_pct']*weight).unsqueeze(0),torch.tensor([weight],device=x.device))
                return result
            win,win_aux=self.log_probability(self.model,x,y)
            lose,lose_aux=self.log_probability(self.model,rejected_x,rejected_y)
            policy=win-lose
            with torch.no_grad():
                baseline=self.log_probability(self.reference,x,y)[0]-self.log_probability(self.reference,rejected_x,rejected_y)[0]
            losses=-F.logsigmoid(self.beta*(policy-baseline))+0.5*(win_aux+lose_aux)
            return losses.sum().reshape(1),torch.ones_like(losses).sum().reshape(1)


def reduced_loss(result):
    total,count=result[:2]
    return total.sum()/count.sum().clamp_min(1)
