python reid_visualization.py \
    --query_dir /data/reid/reid_master/data_cleaned/query \
    --gallery_dir /data/reid/reid_master/data_cleaned/gallery \
    --model_path checkpoints/swin_base_msmt17.pth \
    --config_path configs/msmt17/swin_base.yml \
    --output reid_results_base_msmt17.png \
    --rank 10