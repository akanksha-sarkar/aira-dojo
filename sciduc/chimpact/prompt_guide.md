You are working on a team to write, test, and iterate on Python code to solve the fish detection task.
The environment is installed with the necessary libraries.
Here are some details about the task:
1. You need to detect fish (with bounding boxes and classes).
2. There are three classes of fish for this task: bi-color-damselfish, bluehead-wrasse, and brown-chromis.
3. We do not want to detect other types of fish and we only want to detect fish where we can classify their behaviors.
4. You have access to a construct_sample function for selecting **ONLY {K}% images** for training and validation set.
5. You must produce predictions on the test set of the generated yaml file.
6. You have access to torchvision, OpenCV, and ultralytics.
7. Make sure to import the libraries.
    - ```python
        import cv2
        import torchvision
        import ultralytics
    ```
8. To save time, use epoch 20 as maximum
The construct_sample function is provided below:
import os
import random

def construct_sample(ratio):
    yolo_format_dir = "/share/j_sun/xy468/dts_agent/data/ChimpACT/data/yolo_format"
    random.seed(42)

    # Sample images and labels for each split
    splits = ["train", "val", "test"]
    sampled_files = {}

    for split in splits:
        images = os.listdir(os.path.join(yolo_format_dir, "images", split))
        labels = os.listdir(os.path.join(yolo_format_dir, "labels", split))
        sampled_files[split] = {
            "images": random.sample(images, int(len(images) * ratio)),
            "labels": random.sample(labels, int(len(labels) * ratio))
        }

    # Create sample directory structure
    sample_dir = os.path.join(yolo_format_dir, f"sample_{ratio}")
    yaml_file = os.path.join(sample_dir, f"sample_{ratio}.yaml")

    # Check if sample directory already exists
    if os.path.exists(sample_dir) and os.path.exists(yaml_file):
        print(f"Sample dataset already exists: {yaml_file}")
        exit(0)

    for split in splits:
        os.makedirs(os.path.join(sample_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(sample_dir, "labels", split), exist_ok=True)

    # Create symlinks
    for split in splits:
        for image in sampled_files[split]["images"]:
            src = os.path.join(yolo_format_dir, "images", split, image)
            dst = os.path.join(sample_dir, "images", split, image)
            os.symlink(src, dst)
        
        for label in sampled_files[split]["labels"]:
            src = os.path.join(yolo_format_dir, "labels", split, label)
            dst = os.path.join(sample_dir, "labels", split, label)
            os.symlink(src, dst)

    # Create YAML configuration file
    keypoints = [
        "pelvis", "right_knee", "right_ankle", "left_knee", "left_ankle",
        "neck", "upper_lip", "lower_lip", "right_eye", "left_eye",
        "right_shoulder", "right_elbow", "right_wrist",
        "left_shoulder", "left_elbow", "left_wrist"
    ]

    yaml_content = f"""# ChimpACT Pose Estimation Dataset Configuration
    # Ultralytics YOLO COCO Pose Format
    # Documentation: https://docs.ultralytics.com/datasets/pose/coco-pose/

    path: {sample_dir}
    train: images/train
    val: images/val
    test: images/test

    # Keypoints
    kpt_shape: [16, 3] # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
    flip_idx: [0, 3, 4, 1, 2, 5, 6, 7, 9, 8, 13, 14, 15, 10, 11, 12]

    # Classes
    names:
    0: chimpanzee

    # Keypoint names per class
    kpt_names:
    0:
    """
    yaml_content += "\n".join(f"    - {kp}" for kp in keypoints) + "\n"

    with open(yaml_file, "w") as f:
        f.write(yaml_content)

    print(f"Sample dataset created: {yaml_file}")