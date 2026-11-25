"""Dropless token-choice top-k MoE reference implementation for CPU/CUDA."""
import torch
from torch import nn
from torch.nn import functional as F


def expert_mlp(width,dropout):
    return nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width),nn.Dropout(dropout))


class SparseMoE(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.router=nn.Linear(config.n_embd,config.n_experts,bias=False)
        self.experts=nn.ModuleList([expert_mlp(config.n_embd,config.dropout) for _ in range(config.n_experts)])
        self.top_k=config.experts_per_token
        self.aux_coef,self.z_coef=config.router_aux_loss,config.router_z_loss

    def forward(self,x):
        if x.device.type=='xla':raise ValueError('MoE currently supports CPU/CUDA training, not the TPU/XLA path.')
        flat=x.reshape(-1,x.shape[-1])
        # Router arithmetic remains FP32 even under CUDA mixed precision.
        with torch.autocast(device_type=x.device.type,enabled=False):
            logits=F.linear(flat.float(),self.router.weight.float())
            probabilities=logits.softmax(-1)
            weights,indices=probabilities.topk(self.top_k,dim=-1)
            if self.top_k>1:weights=weights/weights.sum(-1,keepdim=True)
            # Top-1 retains its full softmax probability (Switch-style router gradient).
            frequency=F.one_hot(indices,num_classes=len(self.experts)).float().mean((0,1)).detach()
            balance=len(self.experts)*(frequency*probabilities.mean(0)).sum()
            z=logits.logsumexp(-1).square().mean()
            auxiliary=self.aux_coef*balance+self.z_coef*z
        output=torch.zeros_like(flat)
        for number,expert in enumerate(self.experts):
            tokens,slots=torch.where(indices==number)
            if tokens.numel():
                values=expert(flat.index_select(0,tokens))*weights[tokens,slots,None].to(flat.dtype)
                output=output.index_add(0,tokens,values.to(output.dtype))
        return output.view_as(x),auxiliary
