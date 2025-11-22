import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import torchvision.transforms.functional as TF
import json
import random
from pathlib import Path
import yaml

class ObjectDetectionDataset(Dataset):
    """
    A dataset for object detection in COCO format. This class can also generate 
    the YOLO-format version of the dataset with the function generate_yolo_dataset
    which returns the path to the yaml file.

    Args:
        root_dir: str
            The root directory of the dataset in COCO format (split level).
            Example: "/sciduc/wildfin/data/{split}"
        label_json_name: str
            The name of the label json file.
        transform: callable
            Transformations to apply to the images.
    """
    def __init__(self, data_dir, ann_dir, split, transform=None, yolo_dir="/tmp/yolo"):
        self.data_dir = Path(data_dir) / split
        self.ann_dir = Path(ann_dir)
        self.split = split
        self.transform = transform
        self.yolo_dir = Path(yolo_dir) / split
        ann_path = self.ann_dir / split / "annotations.json"
        assert ann_path.exists(), f"Missing annotation file: {ann_path}"

        self.coco = COCO(str(ann_path))

        # Load categories
        with open(ann_path) as f:
            self.categories = json.load(f)["categories"]

        self.img_ids = self.coco.getImgIds()
        self.yolo_yaml_path = self.generate_yolo_dataset()
        
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.data_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")

        image = TF.to_tensor(image)  # shape: [3, H, W]

        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        target = {}
        category_ids = torch.tensor([ann['category_id'] for ann in anns])
        bboxes = [torch.tensor(ann['bbox'], dtype=torch.float32) for ann in anns]
        bboxes = torch.stack(bboxes) if bboxes else torch.zeros((0, 4))
        target['boxes'] = bboxes
        target['class_labels'] = category_ids
        return image, target, img_id

    def generate_yolo_dataset(self):
        """
        Generate YOLO-format dataset by:
        - Symlinking existing images into yolo_dir/images
        - Writing corresponding YOLO label files into yolo_dir/labels
        """
        print("Building YOLO dataset...")
        yolo_image_dir = self.yolo_dir / "images"
        yolo_label_dir = self.yolo_dir / "labels"
        yolo_image_dir.mkdir(parents=True, exist_ok=True)
        yolo_label_dir.mkdir(parents=True, exist_ok=True)

        for img_id in self.img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = Path(self.data_dir) / img_info["file_name"]
            assert img_path.exists(), f"Missing image: {img_path}"

            # --- Create symlink instead of copying/saving ---
            link_target = yolo_image_dir / img_info["file_name"]
            if not link_target.exists():
                try:
                    os.symlink(img_path.resolve(), link_target)
                except FileExistsError:
                    pass  # already exists
                except OSError as e:
                    print(f"⚠️ Could not symlink {img_path} → {link_target}: {e}")

            # --- Create YOLO label file ---
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            label_lines = []
            for ann in anns:
                category_id = ann["category_id"]
                bbox = ann["bbox"]  # COCO format: [x, y, width, height]
                x_center = (bbox[0] + bbox[2] / 2) / img_info["width"]
                y_center = (bbox[1] + bbox[3] / 2) / img_info["height"]
                width = bbox[2] / img_info["width"]
                height = bbox[3] / img_info["height"]
                label_lines.append(
                    f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )

            label_file_path = yolo_label_dir / f"{Path(img_info['file_name']).stem}.txt"
            with open(label_file_path, "w") as f:
                f.write("\n".join(label_lines))


        # Create YAML config file
        yaml_config = {
            'train': str(yolo_image_dir),
            'val': str(yolo_image_dir),
            'nc': len(self.categories),
            'names': [cat['name'] for cat in self.categories]
        }
        yaml_path = self.yolo_dir / f"{self.split}_data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f)

        return str(yaml_path)

    def get_yolo_dataset(self):
        return self.yolo_yaml_path
    
def collate_fn(batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = []
    targets = []
    img_ids = []

    for sample in batch:
        image, target, img_id = sample
        images.append(image)  # shape: [3, H, W]
        targets.append(target)   # shape: [N, 4], N varies per image
        img_ids.append(img_id)
    images = torch.stack(images).to(device)

    return images, targets, img_ids

if __name__ == "__main__": 
    test_dataset = ObjectDetectionDataset(
        data_dir="/data/data",
        ann_dir="/data/annotations/k100",
        split="train",
        yolo_dir="/tmp/yolo",
    )
    print(f"Dataset size: {len(test_dataset)}")
    for i in range(3):
        image, target, img_id = test_dataset[i]
        print(f"Image {i} - ID: {img_id}, Image shape: {image.shape}, Target: {target}")
    
    print(test_dataset.yolo_yaml_path)