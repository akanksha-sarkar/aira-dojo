# Bucktales

## Dataset Description

Understanding animal behaviour is central to predicting, understanding, and mitigating impacts of natural and anthropogenic changes on animal populations and ecosystems. However, the challenges of acquiring and processing long-term, ecologically relevant data in wild settings have constrained the scope of behavioural research. The increasing availability of Unmanned Aerial Vehicles (UAVs), coupled with advances in machine learning, has opened new opportunities for wildlife monitoring using aerial tracking. However, limited availability of datasets with wild animals in natural habitats has hindered progress in automated computer vision solutions for long-term animal tracking. This dataset is the detection subset of a larger dataset called BuckTales that contains annoatated detections of blackbuck antelopes in the wild. It contains 218 images with corresponding bounding box and class annotations. The images are really large ~(3000 x 5000 pixels). Because the images are taken by drones, the antelopes are very very small. There are six classes to detect (with the focus being on antelopes): Drones, birds, unkown, shadow, bbfemale, and bbmale. The data is in COCO object detection format.  

## Task Description

In this task, you must implement a function to detect fish in coral reef environments. 
You have access to an ObjectDetectionDataset for training with corresponding annotations. You must produce predictions on a validation ObjectDetectionDataset which you will be evaluated on. The dataset can be accessed using src.dataset. No need to reinstantiate it.

The ObjectDetectionDataset class is provided below: 

```python
class ObjectDetectionDataset(Dataset):
    """
    A dataset for object detection in COCO format. This class can also access a  
    the YOLO-format version of the dataset with the function get_yolo_dataset().
    which returns the path to the yaml file.

    Args:
        data_dir: str
            The directory of the raw data.
        ann_dir: str
            The directory of the COCO format annotations.
        split: str
            "train", "val", or "test" splits of the data.
        transform: callable
            Transformations to apply to the images.
    """
    def __init__(self):
        # Truncated for brevity
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
        Generate YOLO-format dataset from COCO dataset by:
        - Symlinking existing images
        - Writing corresponding YOLO label files
        """
        print("Building YOLO dataset...")
        #... Truncated for brevity

    def get_yolo_dataset(self):
        """
            Returns the path to the YOLO-format yaml file.
        """
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
```

## Solution Format

You should implement the following skeleton of code with an optimal solution for fish-detection. 

```python
from src.dataset import collate_fn
from torch.utils.data import DataLoader
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

## Important Notes

- Consider exploring different model architectures and approaches. Different detection frameworks may perform better depending on the characteristics of the dataset.
  - torchvision provides various detection models (e.g., Faster R-CNN, RetinaNet, FCOS)
  - timm offers a collection of image models with various backbones
  - ultralytics provides YOLO-based models (including newer versions like YOLO11, YOLO12) as well as RT-DETR
- Try different approaches and don't limit yourself to one framework.