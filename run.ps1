Write-Output "mmpose実行"
conda activate mmpose
python .\mmpose\run_hrnet.py $Args[0]
python .\mmpose\dark-coco.py
Write-Output "MotionBERT実行"
conda activate motionbert
Set-Location .\MotionBERT
python .\choose_file_from_file_manager.py --input ..\$Args --o ..\output
Set-Location ..