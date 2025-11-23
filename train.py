# train.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import os 
import torch.nn.functional as F 
import torch.nn.utils as torch_utils 

from model import TextEncoder, ObjectVAE, LATENT_DIM
from train_dataset import TrainDataset, collate_fn 
from losses import VAELoss, InfoNCELoss 
from data_utils import OBJECT_FEATURE_DIM, get_or_compute_stats 


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
LOCAL_SBERT_PATH = os.path.join(SCRIPT_DIR, "downloaded-models", "all-MiniLM-L6-v2")
TRAIN_DIR = os.path.join(SCRIPT_DIR, "train")
TEXT_MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "text_encoder.pth")
OBJECT_MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "object_vae.pth")


EPOCHS = 20
BATCH_SIZE = 2048
BASE_LEARNING_RATE = 1e-5 
SBERT_LEARNING_RATE = 1e-6
 
KL_WEIGHT = 0.05          
VAE_WEIGHT = 0.1 
ALIGNMENT_WEIGHT = 1.0    
NUM_WORKERS = 0 
CLIP_GRAD_NORM = 1.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. 获取特征统计数据
    print("Initializing feature statistics...")
    mean, std = get_or_compute_stats(TRAIN_DIR)
    mean = mean.to(device)
    std = std.to(device)
    
    # 2. 数据加载
    print("Initializing Dataset...")
    dataset = TrainDataset(TRAIN_DIR, mean=mean.cpu(), std=std.cpu()) 
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True 
    )
    
    # 3. 初始化模型
    print("Initializing Models...")
    text_encoder = TextEncoder(
        latent_dim=LATENT_DIM, 
        model_name=LOCAL_SBERT_PATH
    ).to(device)
    object_vae = ObjectVAE(input_dim=OBJECT_FEATURE_DIM, latent_dim=LATENT_DIM).to(device)

    
    # 检查是否存在旧的权重文件，如果存在，则加载它们以继续训练
    if os.path.exists(TEXT_MODEL_SAVE_PATH):
        print(f"Loading existing weights from {TEXT_MODEL_SAVE_PATH}")
        try:
            text_encoder.load_state_dict(torch.load(TEXT_MODEL_SAVE_PATH, map_location=device))
            print("Text encoder weights loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load text_encoder weights. Starting from scratch. Error: {e}")
    else:
        print("No text_encoder weights found. Starting from scratch.")

    if os.path.exists(OBJECT_MODEL_SAVE_PATH):
        print(f"Loading existing weights from {OBJECT_MODEL_SAVE_PATH}")
        try:
            object_vae.load_state_dict(torch.load(OBJECT_MODEL_SAVE_PATH, map_location=device))
            print("Object VAE weights loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load object_vae weights. Starting from scratch. Error: {e}")
    else:
        print("No object_vae weights found. Starting from scratch.")
    


    # 4. 优化器
    optimizer = optim.AdamW([
        {'params': object_vae.parameters(), 'lr': BASE_LEARNING_RATE},
        {'params': text_encoder.parameters(), 'lr': SBERT_LEARNING_RATE}
    ])
    
    # 5. 损失函数
    vae_loss_fn = VAELoss(kl_weight=KL_WEIGHT).to(device)
    align_loss_fn = InfoNCELoss(initial_logit_scale=2).to(device)
    
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    print(f"--- Starting Training (KAN Gentle Mode: Grad Clip={CLIP_GRAD_NORM}, Logit Scale=1.0, VAE Weight=0.1) ---")
    
    # 6. 训练循环
    for epoch in range(EPOCHS):
        text_encoder.train()
        object_vae.train()
        
        total_epoch_loss = 0
        total_vae_loss = 0
        total_align_loss = 0
        epoch_avg_max_sim = 0.0
        epoch_avg_mean_sim = 0.0
        
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for (texts, obj_tensors) in progress_bar:
            B, D_obj = obj_tensors.shape
            obj_tensors = obj_tensors.to(device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=(device.type == 'cuda')):
                # 1. VAE 损失
                recon_x, mu_vae, logvar_vae = object_vae(obj_tensors)
                loss_vae, recon_loss, kld_loss = vae_loss_fn(recon_x, obj_tensors, mu_vae, logvar_vae)
                
                # 2. 对齐损失 (InfoNCE)
                text_embeddings = text_encoder(texts) 
                obj_embeddings = mu_vae
                
                loss_align = align_loss_fn(text_embeddings, obj_embeddings)
                
                # 总损失
                total_loss = (VAE_WEIGHT * loss_vae) + (ALIGNMENT_WEIGHT * loss_align) 
            
            # 梯度裁剪
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch_utils.clip_grad_norm_(object_vae.parameters(), CLIP_GRAD_NORM)
            torch_utils.clip_grad_norm_(text_encoder.parameters(), CLIP_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            
            
            with torch.no_grad():
                text_norm = F.normalize(text_embeddings.detach(), p=2, dim=1)
                obj_norm = F.normalize(obj_embeddings.detach(), p=2, dim=1)
                sim_matrix = text_norm @ obj_norm.T
                correct_pair_scores = sim_matrix.diag()
                epoch_avg_max_sim += correct_pair_scores.max().item()
                epoch_avg_mean_sim += correct_pair_scores.mean().item()

            total_epoch_loss += total_loss.item()
            total_vae_loss += loss_vae.item() 
            total_align_loss += loss_align.item()

            num_batches_so_far = progress_bar.n + 1
            progress_bar.set_postfix(
                Loss=f"{total_loss.item():.4f}",
                Ali_L=f"{total_align_loss / num_batches_so_far:.4f}", 
                VAE_L=f"{total_vae_loss / num_batches_so_far:.4f}",
                Max_S=f"{epoch_avg_max_sim / num_batches_so_far:.3f}", 
                Avg_S=f"{epoch_avg_mean_sim / num_batches_so_far:.3f}"
            )
            
        avg_max_sim_final = epoch_avg_max_sim / len(loader)
        avg_mean_sim_final = epoch_avg_mean_sim / len(loader)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {total_epoch_loss / len(loader):.4f}, "
              f"Epoch Avg Max Sim: {avg_max_sim_final:.4f}, "
              f"Epoch Avg Mean Sim: {avg_mean_sim_final:.4f}")

 


    print("--- Training Complete ---")
    print(f"Saving models to {TEXT_MODEL_SAVE_PATH} and {OBJECT_MODEL_SAVE_PATH}")
    torch.save(text_encoder.state_dict(), TEXT_MODEL_SAVE_PATH)
    torch.save(object_vae.state_dict(), OBJECT_MODEL_SAVE_PATH)
    print("Models saved successfully.")


if __name__ == "__main__":
    main()