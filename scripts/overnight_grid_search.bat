@echo off
echo ============================================================
echo  Overnight Grid Search - Plant Segmentation Training
echo  Started: %date% %time%
echo  Estimated duration: ~5.5 hours (6 runs)
echo ============================================================

cd /d C:\Users\ethan\grail\grail_tamucc

echo.
echo [Run 1/6] run02 - Heavy stem weight (50 epochs, lr=0.001)
echo Started: %time%
python scripts/train_segmentation.py --data_dirs data/synthetic_plants --output_dir checkpoints/run02 --epochs 50 --batch_size 8 --lr 0.001 --class_weights 0.3 1.5 6.0 --export_onnx
echo Finished: %time%

echo.
echo [Run 2/6] run03 - Heavy stem weight + finer LR (50 epochs, lr=0.0005)
echo Started: %time%
python scripts/train_segmentation.py --data_dirs data/synthetic_plants --output_dir checkpoints/run03 --epochs 50 --batch_size 8 --lr 0.0005 --class_weights 0.3 1.5 6.0 --export_onnx
echo Finished: %time%

echo.
echo [Run 3/6] run04 - Even heavier stem weight (50 epochs, lr=0.001)
echo Started: %time%
python scripts/train_segmentation.py --data_dirs data/synthetic_plants --output_dir checkpoints/run04 --epochs 50 --batch_size 8 --lr 0.001 --class_weights 0.2 1.5 8.0 --export_onnx
echo Finished: %time%

echo.
echo [Run 4/6] run05 - Aggressive weights + finer LR (50 epochs, lr=0.0005)
echo Started: %time%
python scripts/train_segmentation.py --data_dirs data/synthetic_plants --output_dir checkpoints/run05 --epochs 50 --batch_size 8 --lr 0.0005 --class_weights 0.2 2.0 8.0 --export_onnx
echo Finished: %time%

echo.
echo [Run 5/6] run06 - Slow decay + heavy stem (100 epochs, lr=0.001)
echo Started: %time%
python scripts/train_segmentation.py --data_dirs data/synthetic_plants --output_dir checkpoints/run06 --epochs 100 --batch_size 8 --lr 0.001 --class_weights 0.3 1.5 6.0 --export_onnx
echo Finished: %time%

echo.
echo [Run 6/6] run07 - Slow decay + aggressive weights (100 epochs, lr=0.0005)
echo Started: %time%
python scripts/train_segmentation.py --data_dirs data/synthetic_plants --output_dir checkpoints/run07 --epochs 100 --batch_size 8 --lr 0.0005 --class_weights 0.2 2.0 8.0 --export_onnx
echo Finished: %time%

echo.
echo ============================================================
echo  All runs complete! %date% %time%
echo  Compare results below:
echo ============================================================
echo.

for %%d in (run02 run03 run04 run05 run06 run07) do (
    echo --- %%d ---
    if exist checkpoints\%%d\train_config.yaml (
        type checkpoints\%%d\train_config.yaml
    )
    echo.
)

echo.
echo Check each checkpoints/runXX/ directory for:
echo   - best_model.pth (best checkpoint)
echo   - model.onnx (exported ONNX)
echo   - training_curves.png (loss/mIoU plots)
echo.
pause
