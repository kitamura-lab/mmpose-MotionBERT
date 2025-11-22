import json
import os
import sys

import numpy as np

# --- 1. パラメータ設定 ---
INPUT_NPY_PATH = os.path.join('output', 'coco_keypoints.npy')
INPUT_BBOX_NPY_PATH = os.path.join('output', 'bounding_boxes.npy') 
# 出力ファイル名を COCO フォーマットであることを明記する
OUTPUT_JSON_PATH = os.path.join('output', 'dark_coco.json') 

# --- 2. メイン処理ロジック ---
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_npy_full_path = os.path.join(script_dir, INPUT_NPY_PATH)
    input_bbox_npy_full_path = os.path.join(script_dir, INPUT_BBOX_NPY_PATH) 
    output_json_full_path = os.path.join(script_dir, OUTPUT_JSON_PATH)

    print(f"--- 変換開始 (COCO フォーマット) ---")
    print(f"入力キーポイントファイル: {input_npy_full_path}")
    print(f"入力バウンディングボックスファイル: {input_bbox_npy_full_path}") 
    print(f"出力 JSON ファイル: {output_json_full_path}")

    # ステップ 1: データ読み込み
    if not os.path.exists(input_npy_full_path):
        print(f"エラー: 入力ファイル {input_npy_full_path} が見つかりません!")
        sys.exit(1)
    if not os.path.exists(input_bbox_npy_full_path): 
        print(f"エラー: 入力バウンディングボックスファイル {input_bbox_npy_full_path} が見つかりません!")
        sys.exit(1)
        
    try:
        # COCO フォーマットのキーポイントデータを読み込む
        coco_data = np.load(input_npy_full_path) 
        bbox_data = np.load(input_bbox_npy_full_path) 
    except Exception as e:
        print(f"エラー: .npy ファイルの読み込みに失敗しました - {e}")
        sys.exit(1)
        
    total_frames = coco_data.shape[0]

    if coco_data.shape[0] != bbox_data.shape[0]: 
        print(f"エラー: キーポイント ({coco_data.shape[0]} フレーム) とバウンディングボックス ({bbox_data.shape[0]} フレーム) のフレーム数が一致しません!")
        sys.exit(1)
    
    if coco_data.ndim != 3 or coco_data.shape[1] != 17 or coco_data.shape[2] != 3:
         print(f"エラー: キーポイントファイル {input_npy_full_path} のシェイプが (フレーム数, 17, 3) ではありません。実際: {coco_data.shape}")
         sys.exit(1)
         
    if bbox_data.ndim != 2 or bbox_data.shape[1] != 5:
        print(f"エラー: バウンディングボックスファイル {input_bbox_npy_full_path} のシェイプが (フレーム数, 5) ではありません。実際: {bbox_data.shape}")
        sys.exit(1)
        
    print(f"\n[ステップ 1] COCO データ (シェイプ: {coco_data.shape}) とバウンディングボックスデータ (シェイプ: {bbox_data.shape}) の読み込みに成功しました")

    results_list = []

    # ステップ 2: JSON 構造の構築 (フォーマット変換は不要)
    print(f"[ステップ 2] フレームごとに JSON を構築...")
    num_valid_frames = 0
    for i in range(total_frames):
        # COCO フォーマットのキーポイントを直接使用
        coco_kpts_frame = coco_data[i] # Shape (17, 3)
        bbox_frame = bbox_data[i] # Shape (5,) [x_min, y_min, x_max, y_max, score]
        
        # キーポイントが有効かチェック (少なくとも1点の信頼度が > 0)
        if not np.any(coco_kpts_frame[:, 2] > 1e-6): 
            continue

        # キーポイントを [x0, y0, c0, x1, y1, c1, ...] のリストにフラット化
        keypoints_flat = coco_kpts_frame.flatten().tolist()
        
        # バウンディングボックスのフォーマット変換 [x_min, y_min, x_max, y_max, score] -> [x_min, y_min, width, height]
        x_min, y_min, x_max, y_max, bbox_score = bbox_frame
        width = x_max - x_min
        height = y_max - y_min
        
        # 無効なバウンディングボックスのフレームをスキップ
        if width <= 0 or height <= 0 or bbox_score <= 0:
             continue
        else:
             bbox_for_json = [float(x_min), float(y_min), float(width), float(height)]
             frame_score = float(bbox_score) # bbox のスコアを使用

        # 辞書を作成
        frame_dict = {
            "image_id": f"{i}.jpg", 
            "category_id": 1,        
            "keypoints": keypoints_flat, # COCO フォーマットを直接使用
            "score": frame_score, 
            "box": bbox_for_json,       
            "idx": [0.0] # AlphaPose format placeholder             
        }
        results_list.append(frame_dict)
        num_valid_frames += 1

    print(f"   構築完了 {num_valid_frames} 個の有効フレーム.")

    # ステップ 3: JSON ファイルとして保存
    print(f"[ステップ 3] 結果を JSON ファイルに保存...")
    os.makedirs(os.path.dirname(output_json_full_path), exist_ok=True)
    try:
        # コンパクトフォーマットで保存
        with open(output_json_full_path, 'w') as f:
            json.dump(results_list, f, separators=(',', ':')) 
        print(f"   正常に保存されました: {output_json_full_path}")
    except Exception as e:
        print(f"エラー: JSON ファイルの保存に失敗しました - {e}")
        sys.exit(1)

    print(f"\n--- 変換完了 ---")
    print("これで、新しい COCO フォーマットの JSON ファイルを使用して MotionBERT を実行できます:")
    print(f"cd ../MotionBERT")
    print(f"python infer_wild.py --vid_path ../dark/test_video.mp4 --json_path ../dark/output/coco_results_for_motionbert.json --out_path ../dark/output/motionbert_output --config <あなたの設定>.yaml --evaluate <あなたのモデル>.bin")