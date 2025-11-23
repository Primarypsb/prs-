# model.py

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from libs import KAN, LatentParams, reparameterize 
from data_utils import OBJECT_FEATURE_DIM

LATENT_DIM = 128 

class TextEncoder(nn.Module):
    """
    
    使用预训练的 SentenceTransformer 对文本描述进行编码。
    模型参数被解冻以进行微调。
    """
    def __init__(self, latent_dim=LATENT_DIM, model_name='all-MiniLM-L6-v2'):
        super(TextEncoder, self).__init__()
        
        # 添加 local_files_only=True 来强制离线
        self.model = SentenceTransformer(model_name, local_files_only=True)
        sbert_dim = self.model.get_sentence_embedding_dimension()
        self.projection = nn.Linear(sbert_dim, latent_dim)
        
        # (已解冻 SBERT，允许微调)
            
    def forward(self, sentences: list[str]):
        device = self.projection.weight.device
        embeddings = self.model.encode(
            sentences, 
            convert_to_tensor=True, 
            device=device
        )
        latent_vecs = self.projection(embeddings)
        return latent_vecs

class ObjectVAE(nn.Module):
    """
    (已修改)
    只使用 KAN 构建的对象 VAE。
    """
    def __init__(self, input_dim=OBJECT_FEATURE_DIM, latent_dim=LATENT_DIM):
        super(ObjectVAE, self).__init__()
        
       
        # 移除了USE_KAN_MODEL 开关，硬编码为 KAN
        print(f"--- ObjectVAE: Using KAN ---")

        self.encoder_net = KAN(
            layers_hidden=[input_dim, 64, 32]
        )
        
        self.latent_params = LatentParams(32, latent_dim)
        
        self.decoder_net = KAN(
            layers_hidden=[latent_dim, 64, input_dim]
        )
       

    def encode(self, x):
        hidden = self.encoder_net(x)
        mu, logvar = self.latent_params(hidden)
        return mu, logvar

    def decode(self, z):
        reconstruction = self.decoder_net(z)
        return reconstruction

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar