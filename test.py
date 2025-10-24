import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from pycocotools.coco import COCO

# Configuration
DATA_DIR = "/share/j_sun/ethan/sciduc/wildfin/public"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
NUM_CLASSES = 5  # Including background
BATCH_SIZE = 4
NUM_EPOCHS = 5
LEARNING_RATE = 0.005
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
GRADIENT_CLIP = 1.0


# Augmentations
def augment(image):
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                torch.nn.ModuleList(
                    [
                        transforms.ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                        )
                    ]
                ),
                p=0.3,
            ),
            transforms.RandomResizedCrop(
                (IMAGE_HEIGHT, IMAGE_WIDTH), scale=(0.8, 1.0), ratio=(0.9, 1.1)
            ),
            transforms.ToTensor(),
        ]
    )
    return transform(image)


# Dataset
class FishDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotations, transform=None):
        self.root = root
        self.coco = COCO(annotations)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_data = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.root, img_data["file_name"])
        img = Image.open(path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        num_objs = len(anns)
        boxes = []
        labels = []
        for i in range(num_objs):
            xmin = anns[i]["bbox"][0]
            ymin = anns[i]["bbox"][1]
            xmax = xmin + anns[i]["bbox"][2]
            ymax = ymin + anns[i]["bbox"][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(anns[i]["category_id"])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([img_id])
        area = torch.as_tensor([ann["area"] for ann in anns], dtype=torch.float32)
        iscrowd = torch.as_tensor([ann["iscrowd"] for ann in anns], dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            img = self.transform(img)

        if boxes.shape[0] == 0 and self.root == os.path.join(TRAIN_DIR, "images"):
            return self[np.random.randint(0, len(self))]

        return img, target

    def __len__(self):
        return len(self.ids)


# Model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
model.to(DEVICE)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training
train_dataset = FishDataset(
    os.path.join(TRAIN_DIR, "images"),
    os.path.join(TRAIN_DIR, "train_annotations.json"),
    transform=augment,
)


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return tuple(zip(*batch)) if batch else ([], [])


train_data_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
)


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    for i, (images, targets) in enumerate(data_loader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)

        optimizer.step()

        if i % print_freq == 0:
            print(
                f"Epoch: [{epoch}][{i}/{len(data_loader)}], Loss: {losses.item():.4f}"
            )


for epoch in range(NUM_EPOCHS):
    train_one_epoch(model, optimizer, train_data_loader, DEVICE, epoch, print_freq=100)
    lr_scheduler.step()

# Prediction
test_dataset = FishDataset(
    os.path.join(TEST_DIR, "images"),
    os.path.join(TEST_DIR, "test_metadata.json"),
    transform=transforms.ToTensor(),
)
test_data_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    collate_fn=lambda batch: tuple(zip(*batch)),
)

model.eval()
predictions = []
with torch.no_grad():
    for i, (images, targets) in enumerate(test_data_loader):
        images = [image.to(DEVICE) for image in images]
        outputs = model(images)

        for j, output in enumerate(outputs):
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            image_id = test_dataset.ids[i * len(images) + j]

            for box, score, label in zip(boxes, scores, labels):
                if score > 0.3:  # Confidence threshold
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    predictions.append(
                        {
                            "image_id": image_id,
                            "category_id": int(label),
                            "bbox": [x1, y1, width, height],
                            "score": float(score),
                        }
                    )

# Save submission
with open("submission.json", "w") as f:
    json.dump(predictions, f)

print("Submission saved to submission.json")