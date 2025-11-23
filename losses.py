# losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

class VAELoss(nn.Module):
    """
   
    计算 VAE 的损失 = 重构损失 + KL散度损失
    *全部使用 'mean'  reduction*
    """
    def __init__(self, kl_weight=1.0):
        super(VAELoss, self).__init__()
        self.kl_weight = kl_weight
        
        self.reconstruction_loss = nn.MSELoss(reduction='mean')

    def forward(self, recon_x, x, mu, logvar):
        # 重构损失 (均值)
        recon_loss = self.reconstruction_loss(recon_x, x)
        
      
        # KL 散度损失 (取均值)
    
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        #总损失
        total_loss = recon_loss + self.kl_weight * kld_loss
        
        
        # 返回均值损失
        return total_loss, recon_loss, kld_loss

class InfoNCELoss(nn.Module):
    """
    计算 CLIP 风格的 InfoNCE 对比损失。
    """
    def __init__(self, initial_logit_scale=2):
        super(InfoNCELoss, self).__init__()
        self.logit_scale = initial_logit_scale
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, text_embeddings, obj_embeddings):
        """
        Args:
            text_embeddings (torch.Tensor): [B, D_latent]
            obj_embeddings (torch.Tensor):  [B, D_latent]
        """
        device = text_embeddings.device
        
        text_embed_norm = F.normalize(text_embeddings, p=2, dim=1)
        obj_embed_norm = F.normalize(obj_embeddings, p=2, dim=1)
        
        logits_per_text = self.logit_scale * text_embed_norm @ obj_embed_norm.T
        logits_per_obj = logits_per_text.T # (B, B)

        B = text_embeddings.shape[0]
        labels = torch.arange(B, device=device).long()
        
        loss_text = self.loss_fn(logits_per_text, labels)
        loss_obj = self.loss_fn(logits_per_obj, labels)
        
        total_loss = (loss_text + loss_obj) / 2
        
        return total_loss