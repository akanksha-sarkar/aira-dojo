# Cell Segmentation

## Dataset Description

The study of how cells grow, divide, and interact over time is a core problem in biology. A fundamental prerequisite for many downstream analyses—such as tracking, lineage inference, and morphology measurement—is cell segmentation: identifying which pixels belong to which individual cells in an image. The dataset consists of microscopy images of cells along with corresponding instance-level annotations. Each image may contain multiple cells, and each cell should be assigned a unique instance ID. Background pixels are labeled as 0. The data is exposed through a CellSegDataset abstraction, which provides access to images and instance masks and supports exporting the data into YOLO-compatible segmentation format.

## Task Description

Your goal is to implement a function called cell_segmentation that performs instance segmentation on a validation dataset of cell images.

You must implement the following function:

```python
def cell_segmentation(train_dataset: CellSegDataset, val_dataset: CellSegDataset):
    """
	Input
	•	train_dataset: a CellSegDataset containing training images and instance masks.
	•	val_dataset: a CellSegDataset containing validation images (and possibly masks).

    Returns:
        seg_mask_list: list[np.ndarray] of uint16 instance-id masks, one per image in val_dataset
    """
```

You have access to a CellSegDataset for training with corresponding annotations. You must produce predictions on a validation CellSegDataset which you will be evaluated on.

The CellSegDataset class is provided below: 

```python
class CellSegDataset(Dataset):
    """
    General-purpose segmentation dataset that pairs images and masks by numeric index.

    Returns:
        image: H x W x 3 (RGB)
        mask:  H x W (uint16 instance IDs) OR None if require_masks=False and missing
        meta:  dict with filename info
    """

    def __init__(
        self,
        data_dir,
        ann_dir,
        split: str = "train",
        require_masks: bool = True,
        image_glob: str = "*.tif",
        mask_glob: str = "*.tif",
    ):
        self.data_dir = Path(data_dir)
        self.ann_dir = Path(ann_dir)
        self.split = split
        self.require_masks = require_masks

        self.img_dir = self.data_dir
        self.mask_dir = self.ann_dir

        img_paths = sorted(self.img_dir.glob(image_glob))
        if not img_paths:
            raise RuntimeError(f"No images found in {self.img_dir} (glob={image_glob})")

        mask_paths = sorted(self.mask_dir.glob(mask_glob))
        if not mask_paths and require_masks:
            raise RuntimeError(f"No masks found in {self.mask_dir} (glob={mask_glob})")

        # Build index->path maps
        img_by_idx: Dict[int, Path] = {}
        for p in img_paths:
            idx = _extract_index(p.stem)
            if idx is None:
                continue
            img_by_idx[idx] = p

        mask_by_idx: Dict[int, Path] = {}
        for p in mask_paths:
            idx = _extract_index(p.stem)
            if idx is None:
                continue
            mask_by_idx[idx] = p

        if not img_by_idx:
            raise RuntimeError(
                f"Found images in {self.img_dir} but none matched an index pattern (e.g., ...0000.tif)."
            )

        self.pairs: List[Tuple[int, Path, Optional[Path]]] = []
        for idx in sorted(img_by_idx.keys()):
            img_p = img_by_idx[idx]
            m_p = mask_by_idx.get(idx, None)

            if self.require_masks and m_p is None:
                # Hard fail early so you don't discover this deep in training.
                raise RuntimeError(
                    f"Missing mask for image idx={idx}: image={img_p.name}. "
                    f"Expected a mask with the same trailing index in {self.mask_dir} "
                    f"(e.g., man_seg{idx:04d}.tif or mask{idx:04d}.tif)."
                )

            self.pairs.append((idx, img_p, m_p))

        if not self.pairs:
            raise RuntimeError("No (image, mask) pairs could be formed.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        idx, img_path, mask_path = self.pairs[i]

        # --- Load image ---
        img = _read_tif_any(img_path)
        img = _as_rgb(img)

        # --- Load mask ---
        mask = None
        if mask_path is not None:
            m = _read_tif_any(mask_path)
            if m.ndim == 3: # Some GT masks can be saved as HxWx1
                m = m[..., 0]
            mask = m.astype(np.uint16)

        meta = {
            "index": idx,
            "image_id": img_path.stem,
            "image_path": str(img_path),
            "mask_path": str(mask_path) if mask_path is not None else None,
        }

        return img, mask, meta

    def generate_yolo_dataset(self, yolo_dir: str | Path, *, class_id: int = 0, class_name: str = "cell", write_empty_labels: bool = True) -> str:
        """
        Generate Ultralytics YOLOv8-seg dataset for this dataset's split.

        Writes:
            yolo_dir/
            images/<split>/*.png
            labels/<split>/*.txt
            data.yaml

        Returns:
            str path to yolo_dir/data.yaml
        """
        yolo_dir = Path(yolo_dir)
        split = self.split

        print(f"Building YOLO-seg dataset at {yolo_dir} (split={split})...")

        yolo_image_dir = yolo_dir / "images" / split
        yolo_label_dir = yolo_dir / "labels" / split
        yolo_image_dir.mkdir(parents=True, exist_ok=True)
        yolo_label_dir.mkdir(parents=True, exist_ok=True)

        num_written = 0

        for idx, img_path, mask_path in self.pairs:
            assert img_path.exists(), f"Missing image: {img_path}"

            # ---------- IMAGE EXPORT (force 3-channel PNG) ----------
            img = _read_tif_any(img_path)
            img = _as_rgb(img)

            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)

            out_img_name = f"{img_path.stem}.png"
            out_img_path = yolo_image_dir / out_img_name

            if not out_img_path.exists():
                cv2.imwrite(str(out_img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # ---------- LABEL GENERATION ----------
            label_path = yolo_label_dir / f"{img_path.stem}.txt"

            if mask_path is not None and Path(mask_path).exists():
                iou, dice = mask_to_yolo_and_iou(
                    mask_path=mask_path,
                    txt_path=label_path,
                    class_id=class_id,
                )
            else:
                if write_empty_labels:
                    label_path.touch()
                else:
                    continue

            num_written += 1

        # ---------- WRITE data.yaml ----------
        data_yaml_path = yolo_dir / "data.yaml"
        yaml_config = {
            "path": str(yolo_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {class_id: class_name},
        }

        if data_yaml_path.exists():
            try:
                with open(data_yaml_path, "r") as f:
                    existing = yaml.safe_load(f) or {}
            except Exception:
                existing = {}
            existing.update(yaml_config)
            yaml_config = existing

        with open(data_yaml_path, "w") as f:
            yaml.safe_dump(yaml_config, f, sort_keys=False)

        self.yolo_dir = yolo_dir
        self.yolo_yaml_path = str(data_yaml_path)

        print(f"Done. Wrote {num_written} samples to split={split}.")
        return str(data_yaml_path)


    def get_yolo_dataset(self) -> str:
        if not hasattr(self, "yolo_yaml_path"):
            raise RuntimeError("YOLO dataset has not been generated yet. Call generate_yolo_dataset(...) first.")
        return self.yolo_yaml_path

def collate_fn(batch):
    imgs, masks, metas = zip(*batch)
    return list(imgs), list(masks), list(metas)
```