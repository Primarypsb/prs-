# data_utils.py

import json
import torch
import torch.nn.functional as F
import numpy as np
import ast
import os     
import glob   
from tqdm import tqdm 


NUMERIC_FEATURE_INDICES = list(range(1, 15))
COLOR_MAP = {
    'white': 0, 'black': 1, 'red': 2, 'silver-grey': 3,
    'yellow-orange': 4, 'dark-brown': 5, 'bus': 6, 'unknown': 99
}
OBJECT_FEATURE_DIM = len(NUMERIC_FEATURE_INDICES) + 1

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
STATS_FILE_PATH = os.path.join(SCRIPT_DIR, 'feature_stats.pth')



def _parse_features(parts: list, color_index: int):
    try:
        numeric_features = [float(parts[i]) for i in NUMERIC_FEATURE_INDICES]
    except (ValueError, IndexError) as e:
        print(f"Error parsing numeric features from parts: {parts}\nError: {e}")
        return None
    try:
        color_str = parts[color_index]
    except IndexError:
        color_str = 'unknown'
    color_int = COLOR_MAP.get(color_str, COLOR_MAP['unknown'])
    all_features = numeric_features + [color_int]
    features_tensor = torch.tensor(all_features, dtype=torch.float32)
    return features_tensor.unsqueeze(0)


def parse_object_string(line: str):
    parts = line.split(' ')
    if len(parts) == 16:
         return _parse_features(parts, color_index=15)
    elif len(parts) == 17:
         parts_without_score = parts[:15] + [parts[16]]
         return _parse_features(parts_without_score, color_index=15)
    else:
        print(f"Unexpected object string length: {len(parts)} in line: {line}")
        return None


def parse_train_object_string(line_str: str):
    try:
        parts = ast.literal_eval(line_str)
    except Exception as e:
        print(f"Error literal_eval on line: {line_str}\nError: {e}")
        return None
    return _parse_features(parts, color_index=15)


def load_json_data(json_path: str):
    with open(json_path, 'r') as f:
        data = json.load(f)
    public_descriptions = data['public_description']
    test_data_lines = data['test_data']
    return public_descriptions, test_data_lines


def get_or_compute_stats(json_dir, force_compute=False):
    """
    
    加载或计算训练集中所有 239k+ 个对象的均值(mean)和标准差(std)。
    """
    if os.path.exists(STATS_FILE_PATH) and not force_compute:
        print(f"Loading pre-computed feature stats from {STATS_FILE_PATH}")
        stats = torch.load(STATS_FILE_PATH)
        return stats['mean'], stats['std']

    print(f"Computing feature stats... This may take a few minutes.")
    
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in directory: {json_dir}")

    all_object_tensors = []
    for json_path in tqdm(json_files, desc="Stats Pass 1/2: Parsing JSONs"):
        try:
            with open(json_path, 'r') as f:
                train_data = json.load(f) 
                if isinstance(train_data, list) and len(train_data) > 0:
                    for ann in train_data[0]:
                        if 'label_3' in ann and isinstance(ann['label_3'], list):
                            for obj_str in ann['label_3']:
                                obj_tensor = parse_train_object_string(obj_str)
                                if obj_tensor is not None:
                                    all_object_tensors.append(obj_tensor)
        except Exception as e:
            print(f"Warning: Error parsing {json_path} for stats. Skipping. {e}")

    if not all_object_tensors:
        raise ValueError("No objects found to compute stats.")

    print(f"Stats Pass 2/2: Found {len(all_object_tensors)} total objects. Finding unique...")
    unique_tensors = {}
    for t in all_object_tensors:
        unique_tensors[t.numpy().tobytes()] = t
     
    
    stacked_tensors = torch.cat(list(unique_tensors.values()), dim=0)
    print(f"Computing stats over {stacked_tensors.shape[0]} unique objects.")

    mean = torch.mean(stacked_tensors, dim=0)
    std = torch.std(stacked_tensors, dim=0)
    
    std[std == 0] = 1.0

    print(f"Stats computed. Mean: {mean}")
    print(f"Stats computed. Std: {std}")
    
    torch.save({'mean': mean, 'std': std}, STATS_FILE_PATH)
    print(f"Feature stats saved to {STATS_FILE_PATH}")
    
    return mean, std