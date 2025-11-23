import os
import numpy as np
# import argparse # --vid_path と --json_path の解析は不要になった
from tqdm import tqdm
import imageio
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_wild import WildDetDataset
from lib.utils.vismo import render_and_save

# ファイル選択のために tkinter ライブラリをインポート
import tkinter as tk
from tkinter import filedialog



def parse_args():
    parser = argparse.ArgumentParser()
    # 設定とモデルパスは保持
    # parser.add_argument("--config", type=str, default="configs/pretrain/MB_lite.yaml", help="Path to the config file.")
    parser.add_argument("--config", type=str, default="configs/pose3d/MB_ft_h36m.yaml", help="Path to the config file.")
    # parser.add_argument('-e', '--evaluate', default='checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/latest_epoch-lite.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-e', '--evaluate', default='checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    # -j, -v のコマンドライン引数を削除し、-o も必須ではなくす (select_files で設定するため)
    # parser.add_argument('-o', '--out_path', type=str, default=None, help='output path')
    parser.add_argument('-o', '--out_path', type=str, default='output', help='output path')
    parser.add_argument('--pixel', action='store_true', help='align with pixle coordinates')
    parser.add_argument('--focus', type=int, default=None, help='target person id')
    parser.add_argument('--clip_len', type=int, default=50, help='clip length for network input')
    parser.add_argument('--input', type=str, default=None, help='Path to input video file.')
    
    # ファイルダイアログで選択されたパスを保存するために、2つの新しい属性を追加
    opts = parser.parse_args()
    opts.vid_path = None
    opts.json_path = None
    print(opts.input)
    return opts

def select_files(opts):
    # tkinter を初期化
    root = tk.Tk()
    root.withdraw() # メインウィンドウを非表示

    # --- 1. 動画ファイルの選択 ---
    # args = sys.argv
    if opts.input is not None:
        opts.vid_path = opts.input
    else:
        print("動画ファイル (.mp4, .avi など) を選択してください:")
        # opts.vid_path = "1104-16-0_cutted_1.mp4"
        opts.vid_path = filedialog.askopenfilename(
            title="1/3 動画ファイルを選択",
            filetypes=[("Video files", "*.mp4;*.avi;*.mov"), ("All files", "*.*")]
        )
        if not opts.vid_path:
            print("動画ファイルが選択されませんでした。プログラムを終了します。")
            exit()
    print(f"選択された動画: {opts.vid_path}")

    # --- 2. JSON ファイルの選択 ---
    print("\n2D キーポイント検出結果の JSON ファイル (.json) を選択してください:")
    opts.json_path = "../mmpose/output/dark_coco.json"
    # opts.json_path = filedialog.askopenfilename(
    #     title="2/3 2D キーポイント JSON ファイルを選択",
    #     filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    # )
    if not opts.json_path:
        print("JSON ファイルが選択されませんでした。プログラムを終了します。")
        exit()
    print(f"選択された JSON ファイル: {opts.json_path}")
    
    # JSON ファイル名（拡張子なし）を出力ファイルのプレフィックスとして取得
    json_filename_with_ext = os.path.basename(opts.json_path)
    opts.output_prefix = os.path.splitext(json_filename_with_ext)[0] + '_3D'
    
    # --- 3. 出力フォルダの選択 (追加部分) ---
    # print("\n3D 結果ファイルの出力先フォルダを選択してください:")
    # opts.out_path = "output"
    # # opts.out_path = filedialog.askdirectory(
    # #     title="3/3 出力フォルダを選択"
    # # )
    # if not opts.out_path:
    #     print("出力フォルダが選択されませんでした。プログラムを終了します。")
    #     exit()
    print(f"選択された出力フォルダ: {opts.out_path}")


# --- コード実行開始 ---
opts = parse_args()
select_files(opts) # ファイルとフォルダの選択関数を呼び出し

args = get_config(opts.config)
# print(opts.input)

# ... (モデルのロードと初期化部分は変更なし) ...
model_backbone = load_backbone(args)
if torch.cuda.is_available():
    model_backbone = nn.DataParallel(model_backbone)
    model_backbone = model_backbone.cuda()

print('Loading checkpoint', opts.evaluate)
checkpoint = torch.load(opts.evaluate, map_location=lambda storage, loc: storage)
model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
model_pos = model_backbone
model_pos.eval()

testloader_params = {
          'batch_size': 1,
          'shuffle': False,
          'num_workers': 0,
          'pin_memory': True,
          'persistent_workers': False,
          'drop_last': False
}

# --- 動画情報の取得部分 ---
vid = imageio.get_reader(opts.vid_path,  'ffmpeg')
fps_in = vid.get_meta_data()['fps']
vid_size = vid.get_meta_data()['size']

# opts.out_path は select_files によって有効なパスが保証されているため、os.makedirs は正常に動作するはず
# このコード行の位置は変更不要
os.makedirs(opts.out_path, exist_ok=True) 


if opts.pixel:
    # Keep relative scale with pixel coornidates
    wild_dataset = WildDetDataset(opts.json_path, clip_len=opts.clip_len, vid_size=vid_size, scale_range=None, focus=opts.focus)
else:
    # Scale to [-1,1]
    wild_dataset = WildDetDataset(opts.json_path, clip_len=opts.clip_len, scale_range=[1,1], focus=opts.focus)

test_loader = DataLoader(wild_dataset, **testloader_params)

# ... (推論ループ部分は変更なし) ...
results_all = []
with torch.no_grad():
    for batch_input in tqdm(test_loader):
        N, T = batch_input.shape[:2]
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
        if args.no_conf:
            batch_input = batch_input[:, :, :, :2]
        if args.flip:    
            batch_input_flip = flip_data(batch_input)
            predicted_3d_pos_1 = model_pos(batch_input)
            predicted_3d_pos_flip = model_pos(batch_input_flip)
            predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip) # Flip back
            predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
        else:
            predicted_3d_pos = model_pos(batch_input)
        if args.rootrel:
            predicted_3d_pos[:,:,0,:]=0                    # [N,T,17,3]
        else:
            predicted_3d_pos[:,0,0,2]=0
            pass
        if args.gt_2d:
            predicted_3d_pos[...,:2] = batch_input[...,:2]
        results_all.append(predicted_3d_pos.cpu().numpy())

results_all = np.hstack(results_all)
results_all = np.concatenate(results_all)

# --- 出力ファイル名の変更部分 ---
output_mp4 = f'{opts.out_path}/{opts.output_prefix}.mp4'
output_npy = f'{opts.out_path}/{opts.output_prefix}.npy'

# 可視化動画を保存
render_and_save(results_all, output_mp4, keep_imgs=False, fps=fps_in)

if opts.pixel:
    # Convert to pixel coordinates
    results_all = results_all * (min(vid_size) / 2.0)
    results_all[:,:,:2] = results_all[:,:,:2] + np.array(vid_size) / 2.0
    
# 3D 姿勢データを保存
np.save(output_npy, results_all)

print(f"\n3D 姿勢可視化動画の保存先: {output_mp4}")
print(f"3D 姿勢データファイルの保存先: {output_npy}")