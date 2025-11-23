import torch
import numpy as np
import ipdb
import glob
import os
import io
import math
import random
import json
import pickle
import math
from torch.utils.data import Dataset, DataLoader
from lib.utils.utils_data import crop_scale

def coco2h36m(x):
    '''
        Input: x (T x V x C)  
        COCO: {0-nose 1-Leye 2-Reye 3-Lear 4Rear 5-Lsho 6-Rsho 7-Lelb 8-Relb 9-Lwri 10-Rwri 11-Lhip 12-Rhip 13-Lkne 14-Rkne 15-Lank 16-Rank}
        H36M (MotionBERT):
        0: 'root' (Pelvis), 1: 'rhip', 2: 'rkne', 3: 'rank', 4: 'lhip', 5: 'lkne', 6: 'lank', 7: 'belly' (Thorax), 8: 'neck',
        9: 'nose', 10: 'head', 11: 'lsho', 12: 'lelb', 13: 'lwri', 14: 'rsho', 15: 'relb', 16: 'rwri'
    '''
    # --- ここは新しく追加した関数 ---
    T, V, C = x.shape
    if V != 17:
         raise ValueError("coco2h36m は 17 個のキーポイント (COCO 形式) の入力を想定しています")
         
    y = np.zeros([T, 17, C], dtype=x.dtype) # 出力 H36M 17 点

    # H36M[0] Pelvis = (COCO[11] LHip + COCO[12] RHip) / 2
    y[:,0,:] = (x[:,11,:] + x[:,12,:]) * 0.5
    y[:,0,2] = np.minimum(x[:,11,2], x[:,12,2]) # 信頼度は最小値を取得

    # H36M[1] R_Hip = COCO[12]
    y[:,1,:] = x[:,12,:]
    # H36M[2] R_Knee = COCO[14]
    y[:,2,:] = x[:,14,:]
    # H36M[3] R_Ankle = COCO[16]
    y[:,3,:] = x[:,16,:]
    # H36M[4] L_Hip = COCO[11]
    y[:,4,:] = x[:,11,:]
    # H36M[5] L_Knee = COCO[13]
    y[:,5,:] = x[:,13,:]
    # H36M[6] L_Ankle = COCO[15]
    y[:,6,:] = x[:,15,:]

    # H36M[8] Neck = (COCO[5] LShoulder + COCO[6] RShoulder) / 2
    y[:,8,:] = (x[:,5,:] + x[:,6,:]) * 0.5
    y[:,8,2] = np.minimum(x[:,5,2], x[:,6,2])

    # H36M[7] Thorax (Belly) = (H36M[0] Pelvis + H36M[8] Neck) / 2
    y[:,7,:] = (y[:,0,:] + y[:,8,:]) * 0.5
    y[:,7,2] = np.minimum(y[:,0,2], y[:,8,2])
    
    # H36M[9] Nose = COCO[0]
    y[:,9,:] = x[:,0,:]

    # H36M[10] Head = (COCO[1] L_Eye + COCO[2] R_Eye) / 2 (近似)
    y[:,10,:] = (x[:,1,:] + x[:,2,:]) * 0.5
    y[:,10,2] = np.minimum(x[:,1,2], x[:,2,2])

    # H36M[11] L_Shoulder = COCO[5]
    y[:,11,:] = x[:,5,:]
    # H36M[12] L_Elbow = COCO[7]
    y[:,12,:] = x[:,7,:]
    # H36M[13] L_Wrist = COCO[9]
    y[:,13,:] = x[:,9,:]
    # H36M[14] R_Shoulder = COCO[6]
    y[:,14,:] = x[:,6,:]
    # H36M[15] R_Elbow = COCO[8]
    y[:,15,:] = x[:,8,:]
    # H36M[16] R_Wrist = COCO[10]
    y[:,16,:] = x[:,10,:]
    
    return y
    # --- 新しく追加した関数の終了 ---

def halpe2h36m(x):
    '''
        Input: x (T x V x C)  
       //Halpe 26 body keypoints
    {0,  "Nose"},
    ... (関数の内容は変更なし) ...
    '''
    T, V, C = x.shape
    y = np.zeros([T,17,C])
    y[:,0,:] = x[:,19,:]
    y[:,1,:] = x[:,12,:]
    y[:,2,:] = x[:,14,:]
    y[:,3,:] = x[:,16,:]
    y[:,4,:] = x[:,11,:]
    y[:,5,:] = x[:,13,:]
    y[:,6,:] = x[:,15,:]
    y[:,7,:] = (x[:,18,:] + x[:,19,:]) * 0.5
    y[:,8,:] = x[:,18,:]
    y[:,9,:] = x[:,0,:]
    y[:,10,:] = x[:,17,:]
    y[:,11,:] = x[:,5,:]
    y[:,12,:] = x[:,7,:]
    y[:,13,:] = x[:,9,:]
    y[:,14,:] = x[:,6,:]
    y[:,15,:] = x[:,8,:]
    y[:,16,:] = x[:,10,:]
    return y
    
def read_input(json_path, vid_size, scale_range, focus):
    with open(json_path, "r") as read_file:
        results = json.load(read_file)
    kpts_all = []
    for item in results:
        if focus!=None and item['idx']!=focus:
            continue
        kpts = np.array(item['keypoints']).reshape([-1,3])
        kpts_all.append(kpts)
    kpts_all = np.array(kpts_all)

    # --- ここは修正した部分 ---
    T, V, C = kpts_all.shape
    print(f"検出された入力キーポイント数: {V} 点")

    if V == 26 or V == 136: # Halpe 格式
        print("Halpe 形式として認識、halpe2h36m 変換を実行します...")
        kpts_all = halpe2h36m(kpts_all)
    elif V == 17: # COCO 格式
        print("COCO 17 点形式として認識、coco2h36m 変換を実行します...")
        kpts_all = coco2h36m(kpts_all)
    else:
        raise ValueError(f"サポートされていないキーポイント数: {V}。スクリプトは 17 (COCO) または 26/136 (Halpe) 形式のみをサポートしています。")
    # --- 修正終了 ---
        
    if vid_size:
        w, h = vid_size
        scale = min(w,h) / 2.0
        kpts_all[:,:,:2] = kpts_all[:,:,:2] - np.array([w, h]) / 2.0
        kpts_all[:,:,:2] = kpts_all[:,:,:2] / scale
        motion = kpts_all
    if scale_range:
        motion = crop_scale(kpts_all, scale_range)
    return motion.astype(np.float32)

class WildDetDataset(Dataset):
    def __init__(self, json_path, clip_len=243, vid_size=None, scale_range=None, focus=None):
        self.json_path = json_path
        self.clip_len = clip_len
        self.vid_all = read_input(json_path, vid_size, scale_range, focus)
        
    def __len__(self):
        'Denotes the total number of samples'
        return math.ceil(len(self.vid_all) / self.clip_len)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        st = index*self.clip_len
        end = min((index+1)*self.clip_len, len(self.vid_all))
        return self.vid_all[st:end]