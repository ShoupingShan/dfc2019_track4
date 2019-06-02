#!/bin/bash
source activate shp_tf_18
cd /home/xaserver1/Documents/Contest/shp/Track4
echo "Fold 1"
CUDA_VISIBLE_DEVICES=1 python dfc2019_track4/dfc/train_fold_sift_multi_gpu.py --log_dir=dfc2019_track4/dfc/model_5_fold_gpu_1_wdp --max_epoch=101 --num_gpus=1 --data_dir=dfc2019_track4/data/dfc/train --model=pointSIFT_pointnet --batch_size=5 --fold=1 --existing_model=dfc2019_track4/dfc/model_5_fold_gpu_1_wdp/best_model.ckpt --extra-dims 3 4
echo "Fold 2"
CUDA_VISIBLE_DEVICES=1 python dfc2019_track4/dfc/train_fold_sift_multi_gpu.py --log_dir=dfc2019_track4/dfc/model_5_fold_gpu_1_wdp --max_epoch=201 --num_gpus=1 --data_dir=dfc2019_track4/data/dfc/train --model=pointSIFT_pointnet --batch_size=5 --fold=2 --existing_model=dfc2019_track4/dfc/model_5_fold_gpu_1_wdp/best_model.ckpt --starting_epoch=101 --extra-dims 3 4
echo "Fold 3"
CUDA_VISIBLE_DEVICES=1 python dfc2019_track4/dfc/train_fold_sift_multi_gpu.py --log_dir=dfc2019_track4/dfc/model_5_fold_gpu_1_wdp --max_epoch=301 --num_gpus=1 --data_dir=dfc2019_track4/data/dfc/train --model=pointSIFT_pointnet --batch_size=5 --fold=3 --existing_model=dfc2019_track4/dfc/model_5_fold_gpu_1_wdp/best_model.ckpt --starting_epoch=201 --extra-dims 3 4
echo "Fold 4"
CUDA_VISIBLE_DEVICES=1 python dfc2019_track4/dfc/train_fold_sift_multi_gpu.py --log_dir=dfc2019_track4/dfc/model_5_fold_gpu_1_wdp --max_epoch=401 --num_gpus=1 --data_dir=dfc2019_track4/data/dfc/train --model=pointSIFT_pointnet --batch_size=5 --fold=4 --existing_model=dfc2019_track4/dfc/model_5_fold_gpu_1_wdp/best_model.ckpt --starting_epoch=301 --extra-dims 3 4
echo "Fold 5"
CUDA_VISIBLE_DEVICES=1 python dfc2019_track4/dfc/train_fold_sift_multi_gpu.py --log_dir=dfc2019_track4/dfc/model_5_fold_gpu_1_wdp --max_epoch=501 --num_gpus=1 --data_dir=dfc2019_track4/data/dfc/train --model=pointSIFT_pointnet --batch_size=5 --fold=5 --existing_model=dfc2019_track4/dfc/model_5_fold_gpu_1_wdp/best_model.ckpt --starting_epoch=401 --extra-dims 3 4
echo "Train Done!"
echo "Prediction Start"
CUDA_VISIBLE_DEVICES=1 python dfc2019_track4/dfc/inference.py --model=pointSIFT_pointnet --extra-dims 3 4  --model_path=dfc2019_track4/dfc/model_5_fold_gpu_1_wdp/best_model.ckpt --n_angles=12 --batch_size=5 --input_path=dfc2019_track4/data/dfc/inference_data/in   --output_path=dfc2019_track4/data/dfc/inference_data/out_5_fold_2_ang_12_start_all_wdp_bs_5
echo "Start calculate confusion matrics"
CUDA_VISIBLE_DEVICES=1 python track4-metrics.py -g dfc2019_track4/data/dfc/inference_data/gt -d dfc2019_track4/data/dfc/inference_data/out_5_fold_2_ang_12_start_all_wdp_bs_5 > out_5_fold_2_ang_12_start_all_wdp_bs_5.txt
echo "Inference data Done"
source deactivate