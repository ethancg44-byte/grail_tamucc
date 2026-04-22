@echo off
REM 9-hour sweep: 7 runs testing data split, loss, resolution, LR
REM Est. total: ~9 hours on CUDA (~43s/epoch at 256, longer at higher res)

echo ============================================================
echo   9-Hour Stem IoU Sweep (runs 22-28)
echo   Started: %date% %time%
echo ============================================================

REM ===============================================================
REM PHASE 1: 90/5/5 split runs (reshuffle first)
REM ===============================================================
echo.
echo [SETUP] Reshuffling splits to 90/5/5...
python scripts/reshuffle_splits.py --data_dir data/synthetic_plants --split 0.9 0.05 0.05 --seed 42

REM --- Run 22: Best config + 90/5/5 split ---
echo.
echo [Run 22] LRASPP + CE+Dice + stem_sampling + 90/5/5 split
echo Started: %date% %time%
python scripts/train_segmentation.py ^
    --data_dirs data/synthetic_plants ^
    --output_dir checkpoints/run22_cedice_stemsamp_9055 ^
    --model lraspp ^
    --loss_type ce_dice ^
    --class_weights 0.5 2.0 3.0 ^
    --stem_sampling ^
    --epochs 100 ^
    --batch_size 8 ^
    --lr 0.001 ^
    --aug_level basic ^
    --warmup_epochs 5 ^
    --input_size 256
echo [Run 22] Finished: %date% %time%

REM --- Run 28: CE+Lovász + stem_sampling + 90/5/5 ---
echo.
echo [Run 28] LRASPP + CE+Lovasz + stem_sampling + 90/5/5 split
echo Started: %date% %time%
python scripts/train_segmentation.py ^
    --data_dirs data/synthetic_plants ^
    --output_dir checkpoints/run28_celovasz_stemsamp_9055 ^
    --model lraspp ^
    --loss_type ce_lovasz ^
    --class_weights 0.5 2.0 3.0 ^
    --stem_sampling ^
    --epochs 100 ^
    --batch_size 8 ^
    --lr 0.001 ^
    --aug_level basic ^
    --warmup_epochs 5 ^
    --input_size 256
echo [Run 28] Finished: %date% %time%

REM ===============================================================
REM PHASE 2: Restore 80/10/10 split for remaining runs
REM ===============================================================
echo.
echo [SETUP] Restoring 80/10/10 splits...
python scripts/reshuffle_splits.py --data_dir data/synthetic_plants --split 0.8 0.1 0.1 --seed 42

REM --- Run 23: CE+Lovász on LRASPP (untested combo) ---
echo.
echo [Run 23] LRASPP + CE+Lovasz + stem_sampling
echo Started: %date% %time%
python scripts/train_segmentation.py ^
    --data_dirs data/synthetic_plants ^
    --output_dir checkpoints/run23_celovasz_stemsamp ^
    --model lraspp ^
    --loss_type ce_lovasz ^
    --class_weights 0.5 2.0 3.0 ^
    --stem_sampling ^
    --epochs 100 ^
    --batch_size 8 ^
    --lr 0.001 ^
    --aug_level basic ^
    --warmup_epochs 5 ^
    --input_size 256
echo [Run 23] Finished: %date% %time%

REM --- Run 24: Higher resolution 320x320 ---
echo.
echo [Run 24] LRASPP + CE+Dice + stem_sampling + input_size=320
echo Started: %date% %time%
python scripts/train_segmentation.py ^
    --data_dirs data/synthetic_plants ^
    --output_dir checkpoints/run24_cedice_stemsamp_320 ^
    --model lraspp ^
    --loss_type ce_dice ^
    --class_weights 0.5 2.0 3.0 ^
    --stem_sampling ^
    --epochs 100 ^
    --batch_size 8 ^
    --lr 0.001 ^
    --aug_level basic ^
    --warmup_epochs 5 ^
    --input_size 320
echo [Run 24] Finished: %date% %time%

REM --- Run 25: Higher resolution 384x384 ---
echo.
echo [Run 25] LRASPP + CE+Dice + stem_sampling + input_size=384
echo Started: %date% %time%
python scripts/train_segmentation.py ^
    --data_dirs data/synthetic_plants ^
    --output_dir checkpoints/run25_cedice_stemsamp_384 ^
    --model lraspp ^
    --loss_type ce_dice ^
    --class_weights 0.5 2.0 3.0 ^
    --stem_sampling ^
    --epochs 100 ^
    --batch_size 8 ^
    --lr 0.001 ^
    --aug_level basic ^
    --warmup_epochs 5 ^
    --input_size 384
echo [Run 25] Finished: %date% %time%

REM --- Run 26: Lower LR sweep ---
echo.
echo [Run 26] LRASPP + CE+Dice + stem_sampling + lr=0.0005
echo Started: %date% %time%
python scripts/train_segmentation.py ^
    --data_dirs data/synthetic_plants ^
    --output_dir checkpoints/run26_cedice_stemsamp_lr0005 ^
    --model lraspp ^
    --loss_type ce_dice ^
    --class_weights 0.5 2.0 3.0 ^
    --stem_sampling ^
    --epochs 100 ^
    --batch_size 8 ^
    --lr 0.0005 ^
    --aug_level basic ^
    --warmup_epochs 5 ^
    --input_size 256
echo [Run 26] Finished: %date% %time%

REM --- Run 27: Longer warmup ---
echo.
echo [Run 27] LRASPP + CE+Dice + stem_sampling + warmup=10
echo Started: %date% %time%
python scripts/train_segmentation.py ^
    --data_dirs data/synthetic_plants ^
    --output_dir checkpoints/run27_cedice_stemsamp_warmup10 ^
    --model lraspp ^
    --loss_type ce_dice ^
    --class_weights 0.5 2.0 3.0 ^
    --stem_sampling ^
    --epochs 100 ^
    --batch_size 8 ^
    --lr 0.001 ^
    --aug_level basic ^
    --warmup_epochs 10 ^
    --input_size 256
echo [Run 27] Finished: %date% %time%

echo.
echo ============================================================
echo   All runs completed: %date% %time%
echo ============================================================
pause
