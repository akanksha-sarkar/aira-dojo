# WildFin

## Dataset Description

The study of in-situ marine organism behavior is crucial for ecology, biology, and conversation, which has led to substantial video data collection via field-deployable cameras and divers. However, the lack of such large-scale, publicly available, expert-annotated video datasets capturing these natural behaviors severely limits progress in development of automated solutions. This dataset is the detection subset of a larger dataset on fish behavior / tracking in the wild. It contains 1000+ images with corresponding bounding box and class annotations. The data is provided in COCO object detection format.  

## Task Description

In this task, you must finetune pretrained models to detect fish (with bounding boxes and classes). The categories for the dataset are the following: 
  "categories": [
    {
      "id": 0,
      "name": "fish",
      "supercategory": "none"
    },
    {
      "id": 1,
      "name": "bi-color-damselfish",
      "supercategory": "fish",
    },
    {
      "id": 2,
      "name": "bluehead-wrasse",
      "supercategory": "fish",
    },
    {
      "id": 3,
      "name": "brown-chromis",
      "supercategory": "fish",
    }
  ]
  We do not want to detect other types of fish and we only want to detect fish where we can classify their behaviors. You have access to a training dataset with 1000+ images to finetune pretrained models. You must make predictions on the validation set. Here is a skeleton program that you should implement: 

```python
from src.dataset import collate_fn
from torch.utils.data import DataLoader
import cv2
import torchvision
def finetune(train_dataset, val_dataset):
    """
    This function should produce a list of COCO format detection results for each image in the validation dataset. 
    You can use whatever method you want to produce the detection results. Here is some useful information:
    - You can use the train_dataloader to get the images and targets for the training dataset.
    - You can use the val_dataloader to get the images and targets for the validation dataset.
    Args: 
        train_dataset: ObjectDetectionDataset
        val_dataset: ObjectDetectionDataset
    Returns:
        coco_results: A list of COCO format detection results for each image in the validation dataset. 
    """
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    coco_results = []
    return coco_results
```

You have access to an ObjectDetectionDataset for training with corresponding annotations. You must produce predictions on a validation ObjectDetectionDataset which you will be evaluated on.

The ObjectDetectionDataset class is provided below: 

```python
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
    A dataset for object detection in COCO format.

    Args:
        root_dir: str
            The root directory of the dataset in COCO format (split level).
            Example: "/share/j_sun/ethan/{dataset_name}/coco_format/{split}/"
        label_json_name: str
            The name of the label json file.
        transform: callable
            Transformations to apply to the images.
    """
    def __init__(self, root_dir, label_json_name=None, transform=None):
        self.root_dir = root_dir
        self.split = os.path.basename(os.path.normpath(root_dir))
        without_split = os.path.dirname(os.path.normpath(root_dir))  
        self.dataset_dir = os.path.dirname(os.path.normpath(without_split))  
        if label_json_name is None:
            label_json_name = [f for f in os.listdir(root_dir) if f.endswith(".json")][0]
        self.label_json_name = label_json_name
        self.coco = COCO(os.path.join(root_dir, label_json_name))

        with open(os.path.join(root_dir, label_json_name)) as f:
            categories = json.load(f)["categories"]
        self.categories = categories

        self.img_ids = self.coco.getImgIds()
        
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
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
```

## Important Notes

- Remember that the training annotations and submission format should have COCO bounding box format which is [x,y,w,h]. 
- Use the testing metadata to assign your predictions to the correct image ID's.
- **Try to use a pre-trained torchvision model for this task.**
    - **IMPORTANT: You must convert training data to xyxy format for torchvision models!**