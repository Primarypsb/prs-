# losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

class VAELoss(nn.Module):
    """
    计算 VAE 的损失 = 重构损失 + KL散度损失
    """
    def __init__(self, kl_weight=1.0):
        super(VAELoss, self).__init__()
        self.kl_weight = kl_weight
        self.reconstruction_loss = nn.MSELoss(reduction='mean')

    def forward(self, recon_x, x, mu, logvar):
        recon_loss = self.reconstruction_loss(recon_x, x)
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + self.kl_weight * kld_loss
        return total_loss, recon_loss, kld_loss


class InfoNCELoss(nn.Module):
    """
    
    计算 CLIP 风格的 InfoNCE 对比损失。
    * Logit Scale 现在是固定的 (Static) *
    """
    def __init__(self, static_logit_scale=200): 
        super(InfoNCELoss, self).__init__()
        
        
        self.register_buffer('logit_scale', torch.tensor(static_logit_scale))
        
        
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, text_embeddings, obj_embeddings):
        device = text_embeddings.device
        
        text_embed_norm = F.normalize(text_embeddings, p=2, dim=1)
        obj_embed_norm = F.normalize(obj_embeddings, p=2, dim=1)
        
        # 直接使用固定的 scale
        logits_per_text = self.logit_scale * text_embed_norm @ obj_embed_norm.T
        logits_per_obj = logits_per_text.T 

        B = text_embeddings.shape[0]
        labels = torch.arange(B, device=device).long()
        
        loss_text = self.loss_fn(logits_per_text, labels)
        loss_obj = self.loss_fn(logits_per_obj, labels)
        
        return (loss_text + loss_obj) / 2

class BatchHardTripletLoss(nn.Module):
    """
     批次内困难样本挖掘的三元组损失
    目标：sim(pos) > sim(neg) + margin
    """
    def __init__(self, margin=0.4):
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin
        # 使用 PyTorch 自带的 TripletMarginLoss
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')

    def forward(self, text_embeddings, obj_embeddings):
        """
        自动挖掘每个文本对应的"最难"负样本对象
        """
      
        text_norm = F.normalize(text_embeddings, p=2, dim=1)
        obj_norm = F.normalize(obj_embeddings, p=2, dim=1)

     
        sim_matrix = torch.matmul(text_norm, obj_norm.T)
        
      
        B = text_norm.size(0)
        eye_mask = torch.eye(B, device=text_norm.device).bool()
        
    
        sim_matrix_neg = sim_matrix.clone()
        sim_matrix_neg.masked_fill_(eye_mask, -float('inf'))
        
       
        hard_neg_idx = sim_matrix_neg.argmax(dim=1)
        
  
        hard_neg_objs = obj_norm[hard_neg_idx] 
        
      
        loss = self.loss_fn(text_norm, obj_norm, hard_neg_objs)
        
        return loss
