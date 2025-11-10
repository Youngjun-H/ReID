python -m datasets.car.pipeline_rtsp \
    --rtsp_url rtsp://admin:cubox2024%21@172.16.150.130:554/onvif/media?profile=M1_Profile1 \
    --output_dir /data/reid/data/rtsp_1110_roi \
    --roi_file /data/reid/reid_master/roi.txt \
    --filtering_interval 300 \
    --labeling_interval 600