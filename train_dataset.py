# train_dataset.py

import torch
from torch.utils.data import Dataset
import json
import random
from data_utils import parse_train_object_string 
import os
import glob

#定义我们要抖动的特征索引
#15个特征: [Trunc, Occluded, alpha, bbox(4), dim(3), loc(3), ry, color]
#索引:       0      1        2       3-6      7-9     10-12   13   14
#我们抖动 dim, loc, 和 ry (索引 7 到 13)
JITTER_INDICES = list(range(7, 14)) # 7, 8, 9, 10, 11, 12, 13
JITTER_STRENGTH = 0.02 # 抖动强度(百分比)

class TrainDataset(Dataset):
    
    def __init__(self, json_dir, mean, std):
        super(TrainDataset, self).__init__()
        
        self.mean = mean
        self.std = std
        
        self.data = [] 
        self.all_objects_pool = {}

        print(f"Loading and parsing training data from: {json_dir}")
        
        json_files = glob.glob(os.path.join(json_dir, "*.json"))
        total_files = len(json_files)
        print(f"Found {total_files} training files.")
        
        all_annotations = []
        
        for i, json_path in enumerate(json_files):
            print(f"  [File {i+1}/{total_files}] Processing: {os.path.basename(json_path)}")
            try:
                with open(json_path, 'r') as f:
                    train_data = json.load(f) 
                    if isinstance(train_data, list) and len(train_data) > 0:
                        all_annotations.extend(train_data[0])
                    else:
                        print(f"    Warning: Unexpected data format in {json_path}. Skipping.")
            except Exception as e:
                print(f"    Error processing {json_path}: {e}. Skipping.")

        
        print(f"\nLoaded a total of {len(all_annotations)} annotations.")

        print("Populating object pool...")
        for ann in all_annotations:
            if 'label_3' in ann and isinstance(ann['label_3'], list):
                for obj_str in ann['label_3']:
                    if obj_str not in self.all_objects_pool:
                        obj_tensor = parse_train_object_string(obj_str)
                        if obj_tensor is not None:
                            self.all_objects_pool[obj_str] = obj_tensor

        if not self.all_objects_pool: 
            raise ValueError("No valid objects were parsed from the training file(s).")
            
        print(f"Populated object pool with {len(self.all_objects_pool.keys())} unique objects.")

    
        for ann in all_annotations:
            text = ann.get('public_description')
            if 'label_3' in ann and isinstance(ann['label_3'], list):
                positive_obj_strings = [s for s in ann['label_3'] if s in self.all_objects_pool]
            
            if text and positive_obj_strings:
                for obj_str in positive_obj_strings:
                    self.data.append((text, obj_str))
                
        print(f"Created {len(self.data)} training samples (text-object pairs).")
        print("--- Dataset initialization complete. ---")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
       
        text, positive_key = self.data[idx]
        
        # 获取原始张量
        # 我们 .clone() 以确保抖动不会修改池中的原始张量
        raw_tensor = self.all_objects_pool[positive_key].clone().squeeze(0) # [15]
        
        #2. 数据增强 - 抖动 
        # 仅对我们选择的特征（dim, loc, ry）添加噪声
        features_to_jitter = raw_tensor[JITTER_INDICES]
        noise = (torch.randn_like(features_to_jitter) * JITTER_STRENGTH)
        raw_tensor[JITTER_INDICES] = features_to_jitter + noise
        
        #特征标准化 
        standardized_tensor = (raw_tensor - self.mean) / self.std
        
        return text, standardized_tensor

def collate_fn(batch):
    texts = [item[0] for item in batch]
    obj_tensors = torch.stack([item[1] for item in batch]) # [B, D_obj]
    return texts, obj_tensors