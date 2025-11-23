import os
import sys  # sys.exit のために sys をインポート
import time

import numpy as np

from mmpose.apis import MMPoseInferencer

args = sys.argv

# --- 1. 定数の定義 ---

# 動画ファイルパス (このスクリプトからの相対パス)
# VIDEO_PATH = '1104-16-0_cutted_1.mp4'
if len(args) >= 2:
    VIDEO_PATH = args[1]
else:
    VIDEO_PATH = 'input/zhang.mp4'

# 現在のスクリプトが存在するディレクトリを自動取得
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ダウンロードしたファイル名と完全に一致することを確認 (ハイフン '-' を使用)
CONFIG_FILE_NAME = 'td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288.py'
WEIGHTS_FILE_NAME = 'td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288-39c3c381_20220916.pth'

# スクリプトディレクトリとファイル名を結合して絶対パスを作成
CONFIG_FILE_PATH = os.path.join(SCRIPT_DIR, CONFIG_FILE_NAME)
WEIGHTS_FILE_PATH = os.path.join(SCRIPT_DIR, WEIGHTS_FILE_NAME)

# 出力ディレクトリを定義 (このスクリプトからの相対パス)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
OUTPUT_KEYPOINT_NUMPY_FILE = os.path.join(OUTPUT_DIR, 'coco_keypoints.npy')
OUTPUT_BBOX_NUMPY_FILE = os.path.join(OUTPUT_DIR, 'bounding_boxes.npy') # バウンディングボックスのファイル名

# 出力ディレクトリの存在を確認 (なければ作成)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. モデルの初期化 ---
print(f"--- MMPose 推論器を初期化中 ---")
print(f"スクリプトのディレクトリ: {SCRIPT_DIR}")
print(f"モデル設定: {CONFIG_FILE_PATH}")
print(f"モデルの重み: {WEIGHTS_FILE_PATH}")

# ファイルの存在を確認
if not os.path.exists(CONFIG_FILE_PATH):
    print(f"エラー: 設定ファイルが見つかりません! {CONFIG_FILE_NAME} がスクリプトディレクトリにあるか確認してください。")
    sys.exit(1) # sys.exit() で終了
if not os.path.exists(WEIGHTS_FILE_PATH):
    print(f"エラー: 重みファイルが見つかりません! {WEIGHTS_FILE_NAME} がスクリプトディレクトリにあるか確認してください。")
    sys.exit(1) # sys.exit() で終了

# 推論器を初期化
inferencer = MMPoseInferencer(
    pose2d=CONFIG_FILE_PATH,
    pose2d_weights=WEIGHTS_FILE_PATH,
    device='cuda:0' # GPU の使用を明示的に指定。GPUがない場合やメモリ不足の場合は 'cpu' に変更
)
print("--- モデルのロードが完了しました ---")

# --- 3. 動画の推論を実行 (および所要時間を計算) ---
print(f"\n--- 動画を処理中: {VIDEO_PATH} ---")
video_full_path = os.path.join(SCRIPT_DIR, VIDEO_PATH)
if not os.path.exists(video_full_path):
    print(f"エラー: 動画ファイルが見つかりません! {VIDEO_PATH} がスクリプトディレクトリにあるか確認してください。")
    sys.exit(1) # sys.exit() で終了

start_time = time.time()

result_generator = inferencer(
    video_full_path, 
    vis_out_dir=OUTPUT_DIR, # MMPose は自動的に vis サブディレクトリを作成します
    show=False
)
results = [result for result in result_generator] # ジェネレータを消費

end_time = time.time()
total_duration = end_time - start_time
num_frames = len(results)
fps = num_frames / total_duration if total_duration > 0 else 0

print(f"--- 動画処理が完了しました! ---")
print(f"    総所要時間: {total_duration:.2f} 秒")
print(f"    総フレーム数: {num_frames} フレーム")
print(f"    平均フレームレート (FPS): {fps:.2f} フレーム/秒")
vis_output_path = os.path.join(OUTPUT_DIR, 'vis', os.path.basename(VIDEO_PATH)) # デフォルトで vis サブディレクトリに保存されます
print(f"    可視化結果の保存先 (推定): {vis_output_path} (または output/ ディレクトリ内)")


# --- 4. COCO キーポイントとバウンディングボックスの抽出 ---
print(f"\n--- キーポイントとバウンディングボックスのデータを抽出中 ---")
coco_keypoints_sequence = []
bboxes_sequence = [] # バウンディングボックスを保存するため

for frame_idx, frame_result in enumerate(results):
    # ネスト構造の処理: frame_result['predictions'] は [[person1_dict, ...]]
    if len(frame_result['predictions']) > 0:
        person_list = frame_result['predictions'][0]
        if len(person_list) > 0:
            # デフォルトでは、最初に検出された人物のみを取得
            prediction_dict = person_list[0] 
            
            # --- キーポイントの抽出 (COCO フォーマット, ピクセル座標) ---
            keypoints = np.array(prediction_dict['keypoints'])
            scores = np.array(prediction_dict['keypoint_scores'])
            frame_kpts = np.hstack([keypoints, scores[:, np.newaxis]])
            
            # --- バウンディングボックスの抽出 (スコアを含む計5つの値であることを確認) ---
            bbox_full = np.zeros(5) # 0で初期化
            valid_bbox = False
            if 'bbox' in prediction_dict and len(prediction_dict['bbox']) > 0:
                # bbox リストには通常 [x_min, y_min, x_max, y_max] または [x_min, y_min, x_max, y_max, score] が含まれる
                bbox_raw = prediction_dict['bbox'][0] 
                
                # スコアは通常 'bbox_score' フィールドにあり、リストまたはスカラーの場合がある
                if 'bbox_score' in prediction_dict:
                    # .get() を使用してデフォルト値 0.0 を提供 (スコアがない場合に備えて)
                    bbox_score_val = float(np.array(prediction_dict.get('bbox_score', [0.0])).flatten()[0])
                elif len(bbox_raw) == 5: # スコアが bbox リストの5番目の要素として直接存在する場合
                     bbox_score_val = float(bbox_raw[4])
                else: # スコアが見つからない場合はデフォルト値を設定
                    print(f"警告: {frame_idx} フレーム目で bbox_score が見つかりません。デフォルト値 0.0 を使用します。")
                    bbox_score_val = 0.0
                
                # 最初の4つの座標値のみを取得し、それらが浮動小数点数であることを確認
                bbox_coords = np.array(bbox_raw[:4], dtype=float)
                
                # 座標の有効性をチェック
                if bbox_coords[2] > bbox_coords[0] and bbox_coords[3] > bbox_coords[1]:
                     # 4つの座標と1つのスコアを5要素の numpy 配列に結合
                    bbox_full = np.append(bbox_coords, bbox_score_val)
                    valid_bbox = True
                else:
                    print(f"警告: {frame_idx} フレーム目のバウンディングボックス座標が無効です ({bbox_coords})。0 で埋めます。")
                    
            else: # 'bbox' フィールドが見つかりません
                print(f"警告: {frame_idx} フレーム目にバウンディングボックスフィールドが見つかりません。0 で埋めます。")

            # bbox が有効な場合のみキーポイントと bbox を追加し、それ以外は 0 で埋める
            if valid_bbox:
                coco_keypoints_sequence.append(frame_kpts)
                bboxes_sequence.append(bbox_full)
            else:
                coco_keypoints_sequence.append(np.zeros((17, 3)))
                bboxes_sequence.append(np.zeros(5))

        else: # 内部リストが空です
            print(f"警告: {frame_idx} フレーム目で人物が検出されませんでした (内部リストが空)。0 で埋めます。")
            coco_keypoints_sequence.append(np.zeros((17, 3)))
            bboxes_sequence.append(np.zeros(5))
    else: # 'predictions' リストが空です
        print(f"警告: {frame_idx} フレーム目で人物が検出されませんでした (外部リストが空)。0 で埋めます。")
        coco_keypoints_sequence.append(np.zeros((17, 3)))
        bboxes_sequence.append(np.zeros(5))

# 変換してシェイプを確認
coco_keypoints_np = np.array(coco_keypoints_sequence)
bboxes_np = np.array(bboxes_sequence) 
print(f"    キーポイント配列のシェイプ: {coco_keypoints_np.shape}") # 期待値 (フレーム数, 17, 3)
print(f"    バウンディングボックス配列のシェイプ: {bboxes_np.shape}") # 期待値 (フレーム数, 5)

# --- 5. Numpy ファイルの保存 ---
np.save(OUTPUT_KEYPOINT_NUMPY_FILE, coco_keypoints_np)
np.save(OUTPUT_BBOX_NUMPY_FILE, bboxes_np) 

print(f"\n--- 成功! ---")
print(f"1. COCO キーポイントの Numpy データ保存先: {OUTPUT_KEYPOINT_NUMPY_FILE}")
print(f"2. バウンディングボックス (スコア付き) の Numpy データ保存先: {OUTPUT_BBOX_NUMPY_FILE}") 
print(f"3. 可視化動画の保存先 (推定): {vis_output_path}")