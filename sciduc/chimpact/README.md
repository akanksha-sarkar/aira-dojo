# README
## Test Code optimization
This is tested with weco, not airo-dojo yet.
For weco running: 
```
weco run --source pose_estimation.py \
    --eval-command "python evaluate.py --program pose_estimation.py --data /share/j_sun/xy468/dts_agent/data/ChimpACT/data/yolo_format/sample_0.1/sample_0.1.yaml" \
    --metric METRICS \
    --goal maximize \
    --steps 10 \
    --additional-instructions description.md
```
The processed data with different data samples are already in `/share/j_sun/xy468/dts_agent/data/ChimpACT/data/yolo_format/`.

## Data preprocessing
The `src` folder contains how to construct the dataset or how to convert it to yolo format (it is processed already).

- Loading the dataset from the original data (change the split if needed) and convert it to yolo (already processed)
```
python ./src/load_data.py
    --ann_file /home/xy468/dts_agent/animal_track/ChimpACT/data/ChimpACT_processed/annotations/test.json \
    --data_root /home/xy468/dts_agent/animal_track/ChimpACT/data/ChimpACT_processed/test\
    --img_prefix images \
    --convert_to_yolo \
    --yolo_output_dir /home/xy468/dts_agent/animal_track/ChimpACT/data/ChimpACT_processed/test 
```

- **Construct the samples**:
    - Already construct for 0.01, 0.1, 0.2, 0.5, 0.7, 0.9. Need to change the ratio if you need new ones.
```
python ./src/construct_sample.py --ratio 0.1
```
