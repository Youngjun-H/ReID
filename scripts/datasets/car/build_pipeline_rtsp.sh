# # 단일 스트림 + ROI 필터링 + 간격 조정
# python -m datasets.car.pipeline_rtsp \
#     --rtsp_url rtsp://admin:cubox2024%21@172.16.150.130:554/onvif/media?profile=M1_Profile1 \
#     --output_dir /data/yjhwang/reid/data/rtsp_1120_roi \
#     --roi_file /data/yjhwang/reid/reid_master/roi.txt \
#     --filtering_interval 300 \
#     --labeling_interval 600

# 다중 스트림 (2개 CCTV)
python -m datasets.car.pipeline_rtsp \
    --rtsp_urls rtsp://admin:cubox2024%21@172.16.150.130:554/onvif/media?profile=M1_Profile1 \
                rtsp://admin:cubox2024%21@172.16.150.129:554/onvif/media?profile=M1_Profile1 \
    --output_dir /data/yjhwang/reid/data/rtsp_multi_1120_4 \
    --roi_file /data/yjhwang/reid/reid_master/roi.txt \
    --filtering_interval 300 \
    --labeling_interval 600