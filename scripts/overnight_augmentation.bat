@echo off
echo ============================================================
echo  Strong Augmentation Training - Plant Segmentation
echo  Started: %date% %time%
echo  Estimated duration: ~8.5 hours (3 runs)
echo ============================================================

cd /d C:\Users\ethan\grail\grail_tamucc

echo.
echo [Run 1/3] run14 - CE+Dice, strong aug, 250 epochs, lr=0.001
echo   Weights: 0.5/2.0/3.0 (same as best Run08)
echo Started: %time%
python scripts/train_segmentation.py --data_dirs data/synthetic_plants --output_dir checkpoints/run14 --epochs 250 --batch_size 8 --lr 0.001 --class_weights 0.5 2.0 3.0 --loss_type ce_dice --aug_level strong --warmup_epochs 5 --export_onnx
echo Finished: %time%

echo.
echo [Run 2/3] run15 - CE+Dice, strong aug, 250 epochs, lr=0.0005
echo   Weights: 0.5/2.0/3.0 (lower LR variant)
echo Started: %time%
python scripts/train_segmentation.py --data_dirs data/synthetic_plants --output_dir checkpoints/run15 --epochs 250 --batch_size 8 --lr 0.0005 --class_weights 0.5 2.0 3.0 --loss_type ce_dice --aug_level strong --warmup_epochs 5 --export_onnx
echo Finished: %time%

echo.
echo [Run 3/3] run16 - CE+Dice, strong aug, 200 epochs, lr=0.001
echo   Weights: 0.4/2.0/4.0 (heavier stem weight)
echo Started: %time%
python scripts/train_segmentation.py --data_dirs data/synthetic_plants --output_dir checkpoints/run16 --epochs 200 --batch_size 8 --lr 0.001 --class_weights 0.4 2.0 4.0 --loss_type ce_dice --aug_level strong --warmup_epochs 5 --export_onnx
echo Finished: %time%

echo.
echo ============================================================
echo  All runs complete! %date% %time%
echo  Compare results below:
echo ============================================================
echo.

for %%d in (run14 run15 run16) do (
    echo --- %%d ---
    if exist checkpoints\%%d\training_log.csv (
        echo Last 5 lines of training_log.csv:
        powershell -Command "Get-Content checkpoints\%%d\training_log.csv | Select-Object -Last 5"
    )
    echo.
)

echo.
echo Check each checkpoints/runXX/ directory for:
echo   - best_model.pth (best checkpoint)
echo   - model.onnx (exported ONNX)
echo   - training_curves.png (loss/mIoU plots)
echo   - training_log.csv (full epoch log)
echo.
pause
