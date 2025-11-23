# インストール(Python3.8)

## 0. 前提
```
CUDA12.6
miniconda
```

## 1. 新しい conda 環境の作成を推奨
```
conda create -n mmpose python=3.8 -y
conda activate mmpose
```
## 2. PyTorch のインストール
```
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
## 3. OpenMMLab コア依存関係 (MIM) のインストール
```
pip install -U openmim
```
## 4. mmcv, mmdet, mmpose のインストール
```
mim install mmcv==2.1.0
mim install mmdet==3.2.0
mim install mmpose==1.3.2
```
## 5. 重みと設定ファイルをダウンロード
```
mim download mmpose --config td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288 --dest .
```

# 利用法

## 2D骨格抽出の実行

1. スクリプトの準備: `run_hrnet.py` スクリプトを編集し、入力動画パス（VIDEO-PATH）を設定します。
2. スクリプトの実行:
```
python run_hrnet.py
```
1. 出力の取得: 実行後、以下の3つのファイルがoutputフォルダに生成されます。
* `coco_keypoints.npy` (COCO フォーマットの 2D キーポイントファイル)
* `bounding_boxes.npy` (バウンディングボックスファイル)
* `output_video.mp4` (キーポイントが描画された元動画)

## フォーマット変換

MotionBERT は特定の入力フォーマットを必要とします。 `dark-coco.py` スクリプトを使用して変換します。

1. スクリプトの準備: `dark-coco.py` を開き、前のステップで生成された `coco_keypoints.npy` と `bounding_boxes.npy` のファイルパスをスクリプト内に設定します。
2. 変換の実行:
```
python dark-coco.py
```
3. 出力の取得: スクリプトは `MotionBERT` が必要とするフォーマットの 2D データファイル (例: `dark_coco.json`) を生成します。