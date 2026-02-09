import os
import random
import argparse
import textwrap
import shutil
from pathlib import Path

def copy_extra_dir(src_dir: str, dst_root: str):
    """
    Copy everything inside src_dir directly into dst_root.
    Merge and overwrite files if they already exist.
    """
    src_dir = Path(src_dir)
    dst_root = Path(dst_root)

    if not src_dir.exists():
        raise FileNotFoundError(f"Extra directory not found: {src_dir}")

    # Ensure destination exists
    dst_root.mkdir(parents=True, exist_ok=True)

    for item in src_dir.iterdir():
        dst_item = dst_root / item.name
        if item.is_dir():
            shutil.copytree(item, dst_item, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst_item)



def construct_sample(ratio):
    yolo_format_dir = "/share/j_sun/xy468/dts_agent/data/ChimpACT/data/yolo_format"
    random.seed(42)

    # Sample images and labels for each split
    splits = ["train", "val", "test"]
    sampled_files = {}

    for split in splits:
        images = os.listdir(os.path.join(yolo_format_dir, "images", split))
        labels = os.listdir(os.path.join(yolo_format_dir, "labels", split))
        sample_len = int(len(images) * ratio) if split != "test" else len(images)
        sampled_files[split] = {
            "images": random.sample(images, sample_len),
            "labels": random.sample(labels, sample_len)
        }

    # Create sample directory structure
    sample_dir = os.path.join(f"/share/j_sun/as2637/sciduc/chimpact/k{int((ratio) * 100)}")

    # Copying other files into the directory (program files)
    copy_dir =  os.path.join(f"/share/j_sun/as2637/sciduc/chimpact/k{int((ratio) * 100)}") 
    os.makedirs(copy_dir, exist_ok=True)
    extra_dir = "/home/as2637/sciduc/aira-dojo/sciduc/chimpact"  
    copy_extra_dir(extra_dir, copy_dir)
    yaml_file = os.path.join(sample_dir, f"dataset.yaml")

    # Check if sample directory already exists
    if os.path.exists(sample_dir) and os.path.exists(yaml_file):
        print(f"Sample dataset already exists: {yaml_file}")
        exit(0)

    for split in splits:
        os.makedirs(os.path.join(sample_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(sample_dir, "labels", split), exist_ok=True)

    # Copy files (not symlinks) so they work inside containers
    # Symlinks break in containers when target paths aren't mounted
    for split in splits:
        for image in sampled_files[split]["images"]:
            src = os.path.join(yolo_format_dir, "images", split, image)
            dst = os.path.join(sample_dir, "images", split, image)
            if os.path.exists(dst):
                continue  # Skip if already exists
            shutil.copy2(src, dst)
        
        for label in sampled_files[split]["labels"]:
            src = os.path.join(yolo_format_dir, "labels", split, label)
            dst = os.path.join(sample_dir, "labels", split, label)
            if os.path.exists(dst):
                continue  # Skip if already exists
            shutil.copy2(src, dst)

    # Create YAML configuration file
    keypoints = [
        "pelvis", "right_knee", "right_ankle", "left_knee", "left_ankle",
        "neck", "upper_lip", "lower_lip", "right_eye", "left_eye",
        "right_shoulder", "right_elbow", "right_wrist",
        "left_shoulder", "left_elbow", "left_wrist"
    ]

    yaml_header = textwrap.dedent(f"""
    # ChimpACT Pose Estimation Dataset Configuration
    # Ultralytics YOLO COCO Pose Format
    # Documentation: https://docs.ultralytics.com/datasets/pose/coco-pose/

    path: /work
    train: images/train
    val: images/val
    test: images/test

    # Keypoints
    kpt_shape: [16, 3]
    flip_idx: [0, 3, 4, 1, 2, 5, 6, 7, 9, 8, 13, 14, 15, 10, 11, 12]

    # Classes
    names:
      0: chimpanzee

    # Keypoint names per class
    kpt_names:
      0:
    """)

    # Indent keypoint names two levels (4 spaces) under kpt_names -> 0
    kp_lines = "\n".join(f"        - {kp}" for kp in keypoints) + "\n"
    yaml_content = yaml_header + kp_lines

    with open(yaml_file, "w") as f:
        f.write(yaml_content)

    print(f"Sample dataset created: {yaml_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, default=1, help="Ratio of data to sample")
    args = parser.parse_args()
    construct_sample(args.ratio)
