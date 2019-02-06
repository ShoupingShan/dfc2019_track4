#!/bin/bash
echo "Fold 1"
# python pointnet2/utils/show3d_balls.py 
# python pointnet2/utils/Global_confusion_metrics.py
python dfc/train_pointnet2_sift_multigpu.py --log_dir=dfc/model_5_fold --max_epoch=1 --num_gpus=2 --data_dir=data/dfc/train --model=pointSIFT_pointnet --batch_size=10 --fold=1 --extra-dims 3 4
echo "Fold 2"
python dfc/train_pointnet2_sift_multigpu.py --log_dir=dfc/model_5_fold --max_epoch=2 --num_gpus=2 --data_dir=data/dfc/train --model=pointSIFT_pointnet --batch_size=10 --fold=2 --existing_model=best_model.ckpt --starting_epoch=1 --extra-dims 3 4
echo "Fold 3"
python dfc/train_pointnet2_sift_multigpu.py --log_dir=dfc/model_5_fold --max_epoch=3 --num_gpus=2 --data_dir=data/dfc/train --model=pointSIFT_pointnet --batch_size=10 --fold=3 --existing_model=best_model.ckpt --starting_epoch=2 --extra-dims 3 4
echo "Fold 4"
python dfc/train_pointnet2_sift_multigpu.py --log_dir=dfc/model_5_fold --max_epoch=4 --num_gpus=2 --data_dir=data/dfc/train --model=pointSIFT_pointnet --batch_size=10 --fold=4 --existing_model=best_model.ckpt --starting_epoch=3 --extra-dims 3 4
echo "Fold 5"
python dfc2019_track4/dfc/train_pointnet2_sift_multigpu.py --log_dir=dfc/model_5_fold --max_epoch=5 --num_gpus=2 --data_dir=data/dfc/train --model=pointSIFT_pointnet --batch_size=10 --fold=5 --existing_model=best_model.ckpt --starting_epoch=4 --extra-dims 3 4
echo "Train Done!"
echo "Prediction Start"
python dfc/inference.py --model=pointSIFT_pointnet --extra-dims 3 4  --model_path=dfc/model_3_gpu_new/best_model.ckpt  --input_path=data/dfc/inference_data/in   --output_path=data/dfc/inference_data/out_5_fold_gpu_2
echo "Start calculate confusion matrics"
python track4-metrics.py -g dfc2019_track4/data/dfc/inference_data/gt -d dfc2019_track4/data/dfc/inference_data/out_5_fold_gpu_2 > out_5_fold_gpu_2.txt
