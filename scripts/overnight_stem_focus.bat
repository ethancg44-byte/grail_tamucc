@echo off
REM Overnight stem-focus experiments (runs 20-24)
REM Goal: push stem IoU from 0.554 (run08 baseline) toward 0.70
REM
REM Run schedule:
REM   run20: CE+Tversky (isolate Tversky loss effect)
REM   run21: CE+Dice + stem_sampling (isolate oversampling effect)
REM   run22: CE+Tversky + stem_sampling (combine both)
REM   run23: Best combo on 90/5/5 split (requires reshuffle first)
REM   run24: CE+Lovász + stem_sampling on 90/5/5 (backup experiment)

echo ============================================================
echo   Stem-Focus Experiments (runs 20-24)
echo   Started: %date% %time%
echo ============================================================

REM --- Run 20: CE+Tversky on LRASPP (isolate loss effect) ---
echo.
echo [Run 20] LRASPP + CE+Tversky (alpha=0.7, beta=0.3)
echo Started: %date% %time%
python scripts/train_segmentation.py ^
    --data_dirs data/synthetic_plants ^
    --output_dir checkpoints/run20_lraspp_ce_tversky ^
    --model lraspp ^
    --loss_type ce_tversky ^
    --tversky_alpha 0.7 ^
    --tversky_beta 0.3 ^
    --class_weights 0.5 2.0 3.0 ^
    --epochs 100 ^
    --batch_size 8 ^
    --lr 0.001 ^
    --aug_level basic ^
    --warmup_epochs 5 ^
    --input_size 256
echo [Run 20] Finished: %date% %time%

REM --- Run 21: CE+Dice + stem sampling (isolate oversampling) ---
echo.
echo [Run 21] LRASPP + CE+Dice + stem_sampling
echo Started: %date% %time%
python scripts/train_segmentation.py ^
    --data_dirs data/synthetic_plants ^
    --output_dir checkpoints/run21_lraspp_cedice_stemsamp ^
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
echo [Run 21] Finished: %date% %time%

REM --- Run 22: CE+Tversky + stem sampling (combine both) ---
echo.
echo [Run 22] LRASPP + CE+Tversky + stem_sampling
echo Started: %date% %time%
python scripts/train_segmentation.py ^
    --data_dirs data/synthetic_plants ^
    --output_dir checkpoints/run22_lraspp_cetversky_stemsamp ^
    --model lraspp ^
    --loss_type ce_tversky ^
    --tversky_alpha 0.7 ^
    --tversky_beta 0.3 ^
    --class_weights 0.5 2.0 3.0 ^
    --stem_sampling ^
    --epochs 100 ^
    --batch_size 8 ^
    --lr 0.001 ^
    --aug_level basic ^
    --warmup_epochs 5 ^
    --input_size 256
echo [Run 22] Finished: %date% %time%

REM --- Run 23: Best combo on 90/5/5 split ---
REM NOTE: Run reshuffle_splits.py BEFORE this run!
REM   python scripts/reshuffle_splits.py --data_dir data/synthetic_plants --split 0.9 0.05 0.05
echo.
echo [Run 23] LRASPP + CE+Tversky + stem_sampling + 90/5/5 split
echo Started: %date% %time%
echo NOTE: Make sure you ran reshuffle_splits.py first!
python scripts/train_segmentation.py ^
    --data_dirs data/synthetic_plants ^
    --output_dir checkpoints/run23_lraspp_cetversky_stemsamp_9055 ^
    --model lraspp ^
    --loss_type ce_tversky ^
    --tversky_alpha 0.7 ^
    --tversky_beta 0.3 ^
    --class_weights 0.5 2.0 3.0 ^
    --stem_sampling ^
    --epochs 100 ^
    --batch_size 8 ^
    --lr 0.001 ^
    --aug_level basic ^
    --warmup_epochs 5 ^
    --input_size 256
echo [Run 23] Finished: %date% %time%

REM --- Run 24: CE+Lovász + stem sampling on 90/5/5 (backup) ---
echo.
echo [Run 24] LRASPP + CE+Lovasz + stem_sampling + 90/5/5 split
echo Started: %date% %time%
python scripts/train_segmentation.py ^
    --data_dirs data/synthetic_plants ^
    --output_dir checkpoints/run24_lraspp_celovasz_stemsamp_9055 ^
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
echo [Run 24] Finished: %date% %time%

echo.
echo ============================================================
echo   All runs completed: %date% %time%
echo ============================================================
pause
