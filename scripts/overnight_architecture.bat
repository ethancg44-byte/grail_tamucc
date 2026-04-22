@echo off
echo ============================================================
echo  Architecture + Loss Experiment - Plant Segmentation
echo  Started: %date% %time%
echo  Estimated duration: ~7 hours (3 runs)
echo ============================================================

cd /d C:\Users\ethan\grail\grail_tamucc

echo.
echo [Run 1/3] run17 - LRASPP + Lovasz-Softmax, basic aug, 200 epochs
echo   Tests Lovasz loss on current architecture (loss ceiling)
echo Started: %time%
python scripts/train_segmentation.py --data_dirs data/synthetic_plants --output_dir checkpoints/run17 --epochs 200 --batch_size 8 --lr 0.001 --class_weights 0.5 2.0 3.0 --loss_type lovasz --model lraspp --aug_level basic --warmup_epochs 5 --export_onnx
echo Finished: %time%

echo.
echo [Run 2/3] run18 - DeepLabV3 + CE+Dice, basic aug, 200 epochs
echo   Tests architecture upgrade with proven loss (arch ceiling)
echo Started: %time%
python scripts/train_segmentation.py --data_dirs data/synthetic_plants --output_dir checkpoints/run18 --epochs 200 --batch_size 8 --lr 0.001 --class_weights 0.5 2.0 3.0 --loss_type ce_dice --model deeplabv3 --aug_level basic --warmup_epochs 5 --export_onnx
echo Finished: %time%

echo.
echo [Run 3/3] run19 - DeepLabV3 + Lovasz-Softmax, basic aug, 200 epochs
echo   Combines both improvements (expected best)
echo Started: %time%
python scripts/train_segmentation.py --data_dirs data/synthetic_plants --output_dir checkpoints/run19 --epochs 200 --batch_size 8 --lr 0.001 --class_weights 0.5 2.0 3.0 --loss_type lovasz --model deeplabv3 --aug_level basic --warmup_epochs 5 --export_onnx
echo Finished: %time%

echo.
echo ============================================================
echo  All runs complete! %date% %time%
echo ============================================================
echo.

for %%d in (run17 run18 run19) do (
    echo --- %%d ---
    if exist checkpoints\%%d\training_log.csv (
        echo Last 5 lines of training_log.csv:
        powershell -Command "Get-Content checkpoints\%%d\training_log.csv | Select-Object -Last 5"
    )
    echo.
)

echo Check each checkpoints/runXX/ directory for results.
echo.
pause
