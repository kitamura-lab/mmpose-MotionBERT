# インストール(Python3.7)

## 0. 前提
```
CUDA12.6
miniconda
```

## 1. 新しい conda 環境の作成を推奨
```
conda create -n motionbert python=3.7 -y
conda activate motionbert
```
## 2. パッケージのインストール
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
pip install pyyaml
```

## 3. モデルのインストール

[ここ](https://github.com/Walter0807/MotionBERT?tab=readme-ov-file)より3D Pose (H36M-SH, ft)のモデルbest_epoch.binをダウンロードし，checkpoint\pose3d\FT_MB_lite_MB_ft_h36m_global_liteフォルダに保存する．

# 利用法

## スクリプトの実行:
```
python choose_file_from_file_manager.py
```
スクリプトを実行すると、元動画を選択するよう求められます

## 出力の取得: 
プログラムの実行が完了すると、outputフォルダに姿勢推定動画と 3D キーポイントデータが生成されます。