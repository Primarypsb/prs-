# run_inference.py (模型融合版)

import torch
import torch.nn.functional as F
import json
import os 
import argparse
from tqdm import tqdm
import glob

from model import TextEncoder, ObjectVAE, LATENT_DIM
from data_utils import load_json_data, parse_object_string, OBJECT_FEATURE_DIM, STATS_FILE_PATH


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def run_inference_ensemble(json_path, output_dir, 
                           text_enc_1, obj_vae_1, 
                           text_enc_2, obj_vae_2, 
                           device, mean, std): 
    """
    对单个 JSON 文件运行【模型融合】推理流程。
    """
    
    try:
        public_descriptions, test_data_lines = load_json_data(json_path)
    except Exception as e:
        print(f"Error loading {json_path}. Skipping. Error: {e}")
        return

    num_objects = len(test_data_lines)
    if num_objects == 0:
        return

    with torch.no_grad():
      
        text_emb_1 = F.normalize(text_enc_1(public_descriptions), p=2, dim=1)
        
     
        text_emb_2 = F.normalize(text_enc_2(public_descriptions), p=2, dim=1)
        
        obj_emb_list_1 = []
        obj_emb_list_2 = []
        valid_indices = [] 
        
        for i, line in enumerate(test_data_lines):
            raw_tensor = parse_object_string(line)
            if raw_tensor is None:
                continue
            
            raw_tensor = raw_tensor.to(device)
            
            # 标准化(两个模型使用相同的 mean/std)
            standardized_tensor = (raw_tensor - mean) / std
            
         
            mu_1, _ = obj_vae_1.encode(standardized_tensor)
            obj_emb_list_1.append(mu_1)
            
         
            mu_2, _ = obj_vae_2.encode(standardized_tensor)
            obj_emb_list_2.append(mu_2)
            
            valid_indices.append(i) 
            
        if not obj_emb_list_1:
            return

        # 拼接并归一化对象特征
        obj_emb_1 = F.normalize(torch.cat(obj_emb_list_1, dim=0), p=2, dim=1)
        obj_emb_2 = F.normalize(torch.cat(obj_emb_list_2, dim=0), p=2, dim=1)
        
        # 计算相似度矩阵
        sim_1 = torch.matmul(obj_emb_1, text_emb_1.T)
        sim_2 = torch.matmul(obj_emb_2, text_emb_2.T)
        
     
        
        # 打分权重 
        final_sim = 0.6 * sim_1 + 0.4 * sim_2
        
      
        MATCH_THRESHOLD = 0.42
        
      
       
        if 'printed' not in globals():
            print(f"\n[DEBUG] Model 1 Max: {sim_1.max().item():.4f}")
            print(f"[DEBUG] Model 2 Max: {sim_2.max().item():.4f}")
            print(f"[DEBUG] Ensemble Max: {final_sim.max().item():.4f}")
            print(f"[DEBUG] Threshold used: {MATCH_THRESHOLD}\n")
            globals()['printed'] = True
        
        match_matrix = (final_sim > MATCH_THRESHOLD).int()

    output_filename = os.path.basename(json_path).replace('.json', '.txt')
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w') as f:
        full_results = ["0 0 0"] * num_objects
        for i, matrix_row in enumerate(match_matrix):
            original_index = valid_indices[i]
            line_str = f"{matrix_row[0].item()} {matrix_row[1].item()} {matrix_row[2].item()}"
            full_results[original_index] = line_str
        for line_str in full_results:
            f.write(line_str + "\n")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="MonoMulti-3DVG Inference Ensemble Script")
    parser.add_argument('--test_dir', type=str, default="test", help="Directory with input test JSON files.")
    parser.add_argument('--output_dir', type=str, default="result", help="Directory to save the output .txt files.")
    
  
    parser.add_argument('--text_model_1', type=str, default="text_encoder_avg.pth", help="Path to Model 1 text encoder.")
    parser.add_argument('--object_model_1', type=str, default="object_vae_avg.pth", help="Path to Model 1 object VAE.")
    
   
    parser.add_argument('--text_model_2', type=str, default="text_encoder_max.pth", help="Path to Model 2 text encoder.")
    parser.add_argument('--object_model_2', type=str, default="object_vae_max.pth", help="Path to Model 2 object VAE.")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

 
    print(f"Loading feature stats from {STATS_FILE_PATH}...")
    try:
        stats = torch.load(STATS_FILE_PATH, map_location=device)
        mean = stats['mean'].to(device)
        std = stats['std'].to(device)
    except FileNotFoundError:
        print(f"Error: {STATS_FILE_PATH} not found.")
        print("Please run train.py first to generate the feature stats file.")
        exit(1)
    
  
    print("Loading models for Ensemble...")
    try:
        LOCAL_SBERT_PATH = os.path.join(SCRIPT_DIR, "downloaded-models", "all-MiniLM-L6-v2")
        
      
        print(f"Loading Model 1: {args.text_model_1} & {args.object_model_1}")
        path_text_1 = os.path.join(SCRIPT_DIR, args.text_model_1)
        path_obj_1 = os.path.join(SCRIPT_DIR, args.object_model_1)
        
        text_enc_1 = TextEncoder(latent_dim=LATENT_DIM, model_name=LOCAL_SBERT_PATH).to(device)
        text_enc_1.load_state_dict(torch.load(path_text_1, map_location=device))
        text_enc_1.eval()
        
        obj_vae_1 = ObjectVAE(input_dim=OBJECT_FEATURE_DIM, latent_dim=LATENT_DIM).to(device)
        obj_vae_1.load_state_dict(torch.load(path_obj_1, map_location=device))
        obj_vae_1.eval()
        
     
        print(f"Loading Model 2: {args.text_model_2} & {args.object_model_2}")
        path_text_2 = os.path.join(SCRIPT_DIR, args.text_model_2)
        path_obj_2 = os.path.join(SCRIPT_DIR, args.object_model_2)
        
        text_enc_2 = TextEncoder(latent_dim=LATENT_DIM, model_name=LOCAL_SBERT_PATH).to(device)
        text_enc_2.load_state_dict(torch.load(path_text_2, map_location=device))
        text_enc_2.eval()
        
        obj_vae_2 = ObjectVAE(input_dim=OBJECT_FEATURE_DIM, latent_dim=LATENT_DIM).to(device)
        obj_vae_2.load_state_dict(torch.load(path_obj_2, map_location=device))
        obj_vae_2.eval()
        
        print("All models loaded successfully.")
        
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        exit(1)
    except Exception as e: 
        print(f"Error initializing models: {e}")
        exit(1)

 
    test_dir_abs = os.path.join(SCRIPT_DIR, args.test_dir)
    output_dir_abs = os.path.join(SCRIPT_DIR, args.output_dir)
    os.makedirs(output_dir_abs, exist_ok=True)

    test_files = glob.glob(os.path.join(test_dir_abs, "*.json"))
    if not test_files:
        print(f"No .json files found in {test_dir_abs}. Exiting.")
        exit(1)
        
    print(f"Found {len(test_files)} test files. Starting Ensemble Inference...")
    

    for json_path in tqdm(test_files, desc="Processing"):
        run_inference_ensemble(
            json_path=json_path,
            output_dir=output_dir_abs, 
            text_enc_1=text_enc_1, obj_vae_1=obj_vae_1,
            text_enc_2=text_enc_2, obj_vae_2=obj_vae_2,
            device=device,
            mean=mean, std=std    
        )
        
    print(f"Inference complete. Results are in {output_dir_abs}")