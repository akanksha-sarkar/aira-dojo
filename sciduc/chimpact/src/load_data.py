#!/usr/bin/env python3
"""
Standalone data loader for LeipzigChimp pose estimation dataset.

This script demonstrates how to load and work with pose estimation data
without mmengine, mmpose, or mmcv dependencies.
"""

import os
import os.path as osp
import json
from typing import Dict, List, Optional
import argparse
import yaml

import cv2
import numpy as np
import matplotlib.pyplot as plt
from xtcocotools.coco import COCO

# Import our standalone dataset classes
from base_coco_style_dataset import BaseCocoStyleDataset
from leipzigchimp_pose_dataset import LeipzigChimpDataset
from utils import parse_pose_metainfo


def load_leipzigchimp_dataset(ann_file: str, data_root: str = None, img_prefix: str = 'images'):
    """Load LeipzigChimp dataset and explore its contents.
    
    Args:
        ann_file (str): Path to annotation file
        data_root (str, optional): Root directory for data files
        img_prefix (str): Prefix for image directory
    """
    print("=" * 60)
    print("LEIPZIGCHIMP DATASET LOADING")
    print("=" * 60)
    
    # Initialize dataset
    dataset = LeipzigChimpDataset(
        ann_file=ann_file,
        data_root=data_root,
        data_prefix={'img': img_prefix},
        test_mode=False,
        lazy_init=False
    )
    
    print(f"Dataset initialized successfully!")
    print(f"Number of data samples: {len(dataset)}")
    print(f"Metainfo keys: {list(dataset.metainfo.keys())}")
    
    # Print keypoint information
    if 'keypoint_id2name' in dataset.metainfo:
        print(f"\nKeypoint information:")
        for kpt_id, kpt_name in dataset.metainfo['keypoint_id2name'].items():
            print(f"  {kpt_id}: {kpt_name}")
    
    # Load first few samples
    print(f"\nLoading first 3 data samples...")
    for i in range(min(3, len(dataset))):
        data_info = dataset.get_data_info(i)
        print(f"\nSample {i}:")
        print(f"  Image ID: {data_info['img_id']}")
        print(f"  Image path: {data_info['img_path']}")
        print(f"  Bbox: {data_info['bbox']}")
        print(f"  Number of keypoints: {data_info['num_keypoints']}")
        print(f"  Keypoints shape: {data_info['keypoints'].shape}")
        print(f"  Keypoints visible shape: {data_info['keypoints_visible'].shape}")
    
    return dataset


def convert_to_coco_yaml(dataset, output_path: str, dataset_name: str = 'ChimpACT', 
                         split: str = 'train', data_root: str = None, img_prefix: str = 'images'):
    """Convert dataset to Ultralytics COCO pose estimation YAML format.
    
    Args:
        dataset: Dataset instance
        output_path (str): Path to save the YAML file
        dataset_name (str): Name of the dataset
        split (str): Split name (train/val/test)
        data_root (str): Root directory of the dataset (for path field)
        img_prefix (str): Image directory prefix (relative to data_root)
    """
    print(f"\n{'=' * 60}")
    print(f"CONVERTING TO ULRALYTICS COCO YAML FORMAT")
    print(f"{'=' * 60}")
    
    # Get metainfo
    metainfo = dataset.metainfo
    
    # Get keypoint names in order
    num_keypoints = metainfo['num_keypoints']
    kpt_names_list = []
    for kpt_id in range(num_keypoints):
        kpt_name = metainfo['keypoint_id2name'][kpt_id]
        kpt_names_list.append(kpt_name)
    
    # Get flip indices (indices that should be swapped on horizontal flip)
    # flip_idx[i] = j means keypoint i should be swapped with keypoint j when flipping
    flip_indices = []
    if 'flip_indices' in metainfo:
        # flip_indices maps each keypoint to its flipped counterpart by name or id
        flip_list = metainfo['flip_indices']
        keypoint_name2id = metainfo['keypoint_name2id']
        keypoint_id2name = metainfo['keypoint_id2name']
        
        # Create flip_idx list
        for orig_idx in range(num_keypoints):
            flipped_ref = flip_list[orig_idx]
            
            # Convert to index if it's a name
            if isinstance(flipped_ref, str):
                flipped_idx = keypoint_name2id.get(flipped_ref, orig_idx)
            elif isinstance(flipped_ref, (int, np.integer)):
                flipped_idx = int(flipped_ref)
            else:
                flipped_idx = orig_idx
            
            flip_indices.append(flipped_idx)
    else:
        # Default: no flipping (identity mapping)
        flip_indices = list(range(num_keypoints))
    
    # Determine path structure
    if data_root is None:
        # Try to infer from dataset
        if hasattr(dataset, 'data_root') and dataset.data_root:
            path_base = dataset.data_root
        else:
            path_base = f'data/{dataset_name.lower()}'
    else:
        path_base = data_root
    
    # If data_root contains train/val/test, extract the base path
    # e.g., /path/to/data/train -> /path/to/data, and split = train
    if path_base.endswith('/train') or path_base.endswith('\\train'):
        path = path_base.rsplit(os.sep, 1)[0] if os.sep in path_base else path_base.rsplit('/', 1)[0]
        actual_split = 'train'
    elif path_base.endswith('/val') or path_base.endswith('\\val'):
        path = path_base.rsplit(os.sep, 1)[0] if os.sep in path_base else path_base.rsplit('/', 1)[0]
        actual_split = 'val'
    elif path_base.endswith('/test') or path_base.endswith('\\test'):
        path = path_base.rsplit(os.sep, 1)[0] if os.sep in path_base else path_base.rsplit('/', 1)[0]
        actual_split = 'test'
    else:
        path = path_base
        actual_split = split
    
    # Determine train/val/test paths relative to path
    # Format: {split}/images (relative to base path)
    train_path = f"train/{img_prefix}"
    val_path = f"val/{img_prefix}"
    test_path = f"test/{img_prefix}"  # Include test path in YAML (user can comment it out if not needed)
    
    # Create YAML structure in Ultralytics format
    yaml_content = f"""# {dataset_name} Pose Estimation Dataset Configuration
# Ultralytics YOLO COCO Pose Format
# Documentation: https://docs.ultralytics.com/datasets/pose/coco-pose/

# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: {path} # dataset root dir
train: {train_path} # train images (relative to 'path')
val: {val_path} # val images (relative to 'path')"""
    
    yaml_content += f"\ntest: {test_path} # test images (optional)"
    
    yaml_content += f"""

# Keypoints
kpt_shape: [{num_keypoints}, 3] # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
flip_idx: {flip_indices}

# Classes
names:
  0: chimpanzee

# Keypoint names per class
kpt_names:
  0:
"""
    
    # Add keypoint names
    for kpt_name in kpt_names_list:
        yaml_content += f"    - {kpt_name}\n"
    
    # Save to YAML file
    os.makedirs(osp.dirname(output_path) if osp.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✓ Ultralytics COCO YAML config saved to: {output_path}")
    print(f"  - Dataset: {dataset_name}")
    print(f"  - Path: {path}")
    print(f"  - Train: {train_path}")
    print(f"  - Val: {val_path}")
    print(f"  - Keypoints: {num_keypoints}")
    print(f"  - Keypoint shape: [{num_keypoints}, 3]")
    print(f"  - Flip indices: {flip_indices}")
    
    return yaml_content


def convert_to_yolo_pose(dataset, output_dir: str, data_root: str = None, img_prefix: str = 'images'):
    """Convert dataset to YOLO pose format.
    
    YOLO pose format:
    - One .txt label file per image in labels/ folder
    - Each line: class_id x_center y_center width height kpt1_x kpt1_y kpt1_v ...
    - All coordinates normalized (0-1)
    - bbox: center_x, center_y, width, height (normalized)
    - keypoints: x, y, visibility (normalized coordinates)
    - visibility: 0=not labeled, 1=labeled but not visible, 2=labeled and visible
    
    Args:
        dataset: Dataset instance
        output_dir (str): Root directory where labels folder will be created
        data_root (str): Root directory of the dataset
        img_prefix (str): Image directory prefix (relative to data_root)
    """
    print(f"\n{'=' * 60}")
    print(f"CONVERTING TO YOLO POSE FORMAT")
    print(f"{'=' * 60}")
    
    # Determine output structure
    if data_root is None:
        if hasattr(dataset, 'data_root') and dataset.data_root:
            data_root = dataset.data_root
        else:
            data_root = output_dir
    
    # Create labels directory structure
    labels_dir = osp.join(output_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)
    
    num_keypoints = dataset.metainfo['num_keypoints']
    class_id = 0  # Assuming single class (chimpanzee)
    
    processed = 0
    skipped = 0
    
    print(f"Output labels directory: {labels_dir}")
    print(f"Data root: {data_root}")
    print(f"Number of keypoints: {num_keypoints}")
    print(f"Processing {len(dataset)} samples...")
    
    for idx in range(len(dataset)):
        data_info = dataset.get_data_info(idx)
        
        # Get image path and resolve it correctly
        img_path = data_info['img_path']
        
        # If path is relative, make it absolute relative to data_root
        if not osp.isabs(img_path):
            # Join with data_root to get absolute path
            img_path_abs = osp.join(data_root, img_path)
        else:
            img_path_abs = img_path
        
        # Try to get image dimensions
        img_w, img_h = 0, 0
        
        if osp.exists(img_path_abs):
            img = cv2.imread(img_path_abs)
            if img is not None:
                img_h, img_w = img.shape[:2]
            else:
                # Image exists but couldn't be read
                if 'img_shape' in data_info:
                    img_h, img_w = data_info['img_shape']
                elif 'raw_ann_info' in data_info and 'raw_img_info' in data_info:
                    img_info = data_info['raw_img_info']
                    img_w = img_info.get('width', 0)
                    img_h = img_info.get('height', 0)
                else:
                    print(f"Warning: Could not read image or determine size: {img_path_abs}, skipping")
                    skipped += 1
                    continue
        else:
            # Try annotation info first (might have size info even if image doesn't exist)
            if 'raw_ann_info' in data_info and 'raw_img_info' in data_info:
                img_info = data_info['raw_img_info']
                img_w = img_info.get('width', 0)
                img_h = img_info.get('height', 0)
                
                if img_w <= 0 or img_h <= 0:
                    print(f"Warning: Image not found and no valid size info: {img_path_abs} (tried: {img_path}), skipping")
                    skipped += 1
                    continue
            elif 'img_shape' in data_info:
                img_h, img_w = data_info['img_shape']
            else:
                print(f"Warning: Image not found and no size info: {img_path_abs} (tried: {img_path}), skipping")
                skipped += 1
                continue
        
        if img_w <= 0 or img_h <= 0:
            print(f"Warning: Invalid image size {img_w}x{img_h} for {img_path_abs}, skipping")
            skipped += 1
            continue
        
        # Get bbox and keypoints
        bbox = data_info['bbox'][0]  # [x1, y1, x2, y2]
        keypoints = data_info['keypoints'][0]  # [num_keypoints, 2]
        keypoints_visible = data_info['keypoints_visible'][0]  # [num_keypoints]
        
        # Convert bbox from [x1, y1, x2, y2] to [center_x, center_y, width, height] (normalized)
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        
        # Normalize to [0, 1]
        center_x_norm = center_x / img_w
        center_y_norm = center_y / img_h
        width_norm = width / img_w
        height_norm = height / img_h
        
        # Validate bbox
        if width_norm <= 0 or height_norm <= 0 or center_x_norm < 0 or center_y_norm < 0:
            print(f"Warning: Invalid bbox for {img_path_abs}, skipping")
            skipped += 1
            continue
        
        # Prepare keypoint string
        kpt_str_parts = []
        for kpt_idx in range(num_keypoints):
            kpt_x = keypoints[kpt_idx, 0]
            kpt_y = keypoints[kpt_idx, 1]
            visibility = keypoints_visible[kpt_idx]
            
            # Normalize keypoint coordinates
            kpt_x_norm = kpt_x / img_w if kpt_x > 0 else 0.0
            kpt_y_norm = kpt_y / img_h if kpt_y > 0 else 0.0
            
            # Convert visibility: our format uses 0-1, YOLO uses 0,1,2
            # 0 = not labeled, 1 = labeled but not visible, 2 = labeled and visible
            if visibility <= 0:
                vis = 0  # not labeled
            elif visibility < 1:
                vis = 1  # labeled but not visible (occluded)
            else:
                vis = 2  # labeled and visible
            
            kpt_str_parts.append(f"{kpt_x_norm:.6f} {kpt_y_norm:.6f} {vis}")
        
        # Create label file name (mirror image structure)
        # Strategy: Extract the path after img_prefix, preserve subdirectories
        
        # Get relative path from data_root
        try:
            if osp.isabs(img_path) and data_root:
                # Get path relative to data_root
                img_rel_path = osp.relpath(img_path, data_root)
            else:
                img_rel_path = img_path
        except ValueError:
            # Paths on different drives or can't compute relative path
            # Extract path after img_prefix
            if img_prefix in img_path:
                idx = img_path.find(img_prefix) + len(img_prefix)
                img_rel_path = img_path[idx:].lstrip('/\\')
            else:
                img_rel_path = osp.basename(img_path)
        
        # Remove img_prefix from path if it's at the start
        if img_rel_path.startswith(img_prefix):
            img_rel_path = img_rel_path[len(img_prefix):].lstrip('/\\')
        
        # Remove file extension and add .txt
        img_rel_path_no_ext = osp.splitext(img_rel_path)[0]
        
        # Create label path preserving subdirectory structure
        label_path = osp.join(labels_dir, img_rel_path_no_ext + '.txt')
        
        # Create subdirectories if needed
        label_dir = osp.dirname(label_path)
        if label_dir:
            os.makedirs(label_dir, exist_ok=True)
        
        # Write label file (one line per instance in the image)
        # Format: class_id center_x center_y width height kpt1_x kpt1_y kpt1_v ...
        label_line = f"{class_id} {center_x_norm:.6f} {center_y_norm:.6f} {width_norm:.6f} {height_norm:.6f} {' '.join(kpt_str_parts)}\n"
        
        # Check if label file already exists (multiple instances per image)
        if osp.exists(label_path):
            # Append to existing file
            with open(label_path, 'a') as f:
                f.write(label_line)
        else:
            # Create new file
            with open(label_path, 'w') as f:
                f.write(label_line)
        
        processed += 1
        
        if (processed + skipped) % 1000 == 0:
            print(f"  Processed: {processed} | Skipped: {skipped}")
    
    print(f"\n✓ YOLO pose conversion completed!")
    print(f"  - Labels directory: {labels_dir}")
    print(f"  - Processed: {processed} samples")
    print(f"  - Skipped: {skipped} samples")
    print(f"  - Label files created in: {labels_dir}")
    
    return labels_dir


def analyze_dataset_statistics(dataset):
    """Analyze dataset statistics.
    
    Args:
        dataset: Dataset instance
    """
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    
    # Basic statistics
    total_samples = len(dataset)
    print(f"Total samples: {total_samples}")
    
    # Keypoint statistics
    num_keypoints_list = []
    visible_keypoints_list = []
    
    for i in range(min(100, total_samples)):  # Sample first 100 for efficiency
        data_info = dataset.get_data_info(i)
        num_keypoints_list.append(data_info['num_keypoints'])
        visible_keypoints = np.sum(data_info['keypoints_visible'][0] > 0)
        visible_keypoints_list.append(visible_keypoints)
    
    print(f"\nKeypoint statistics (from first {min(100, total_samples)} samples):")
    print(f"  Average number of keypoints: {np.mean(num_keypoints_list):.2f}")
    print(f"  Min keypoints: {np.min(num_keypoints_list)}")
    print(f"  Max keypoints: {np.max(num_keypoints_list)}")
    print(f"  Average visible keypoints: {np.mean(visible_keypoints_list):.2f}")
    
    # Bbox statistics
    bbox_areas = []
    bbox_ratios = []
    
    for i in range(min(100, total_samples)):
        data_info = dataset.get_data_info(i)
        bbox = data_info['bbox'][0]
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        area = w * h
        ratio = w / h if h > 0 else 0
        bbox_areas.append(area)
        bbox_ratios.append(ratio)
    
    print(f"\nBbox statistics:")
    print(f"  Average bbox area: {np.mean(bbox_areas):.2f}")
    print(f"  Average aspect ratio: {np.mean(bbox_ratios):.2f}")


def main():
    """Main function to demonstrate dataset loading."""
    parser = argparse.ArgumentParser(description='Load LeipzigChimp dataset')
    parser.add_argument('--ann_file', type=str, required=True,
                       help='Path to annotation file')
    parser.add_argument('--data_root', type=str, default=None,
                       help='Root directory for data files')
    parser.add_argument('--img_prefix', type=str, default='images',
                       help='Prefix for image directory')
    parser.add_argument('--visualize', type=int, default=None,
                       help='Index of sample to visualize')
    parser.add_argument('--save_viz', type=str, default=None,
                       help='Path to save visualization')
    parser.add_argument('--convert_to_yaml', action='store_true',
                       help='Convert dataset to COCO YAML format')
    parser.add_argument('--yaml_output', type=str, default=None,
                       help='Output path for YAML file (default: dataset_config.yaml)')
    parser.add_argument('--dataset_name', type=str, default='ChimpACT',
                       help='Dataset name for YAML config')
    parser.add_argument('--data_fraction', type=float, default=1.0,
                       help='Fraction of data to use (0.0 to 1.0, default: 1.0 for all data)')
    parser.add_argument('--random_subset', action='store_true',
                       help='Use random sampling instead of first N samples')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for subset sampling (default: 42)')
    parser.add_argument('--convert_to_yolo', action='store_true',
                       help='Convert dataset to YOLO pose format')
    parser.add_argument('--yolo_output_dir', type=str, default=None,
                       help='Output directory for YOLO labels (default: same as data_root)')
    
    args = parser.parse_args()
    
    # Validate data_fraction
    if args.data_fraction <= 0 or args.data_fraction > 1.0:
        parser.error('--data_fraction must be between 0.0 and 1.0')
    
    try:
        # Load dataset
        dataset = load_leipzigchimp_dataset(
            ann_file=args.ann_file,
            data_root=args.data_root,
            img_prefix=args.img_prefix
        )
        
        # Apply data fraction if specified
        if args.data_fraction < 1.0:
            original_size = len(dataset)
            subset_size = int(original_size * args.data_fraction)
            
            # Ensure we have at least 1 sample
            subset_size = max(1, subset_size)
            
            # Select subset
            if args.random_subset:
                # Random sampling
                np.random.seed(args.random_seed)
                indices = np.random.choice(original_size, size=subset_size, replace=False)
                indices = sorted(indices)  # Sort for reproducibility
                dataset._data_list = [dataset._data_list[i] for i in indices]
                sampling_method = "random"
            else:
                # First N samples
                dataset._data_list = dataset._data_list[:subset_size]
                sampling_method = "sequential (first N)"
            
            print(f"\n{'=' * 60}")
            print(f"DATA SUBSET SELECTION")
            print(f"{'=' * 60}")
            print(f"Original dataset size: {original_size}")
            print(f"Fraction selected: {args.data_fraction}")
            print(f"Sampling method: {sampling_method}")
            if args.random_subset:
                print(f"Random seed: {args.random_seed}")
            print(f"Subset size: {len(dataset)}")
            print(f"Using {len(dataset)}/{original_size} samples ({100*len(dataset)/original_size:.1f}%)")
        
        # Analyze statistics
        analyze_dataset_statistics(dataset)
    
        
        # Convert to YAML if requested
        if args.convert_to_yaml:
            # Determine output path
            if args.yaml_output is None:
                # Determine split name from annotation file path
                split_name = 'train'
                if 'val' in args.ann_file.lower():
                    split_name = 'val'
                elif 'test' in args.ann_file.lower():
                    split_name = 'test'
                output_path = f'{args.dataset_name.lower()}_{split_name}_config.yaml'
            else:
                output_path = args.yaml_output
                # Try to determine split from output path if not specified
                split_name = 'train'
                if 'val' in output_path.lower():
                    split_name = 'val'
                elif 'test' in output_path.lower():
                    split_name = 'test'
            
            convert_to_coco_yaml(
                dataset=dataset,
                output_path=output_path,
                dataset_name=args.dataset_name,
                split=split_name,
                data_root=args.data_root,
                img_prefix=args.img_prefix
            )
        
        # Convert to YOLO pose format if requested
        if args.convert_to_yolo:
            # Determine output directory
            if args.yolo_output_dir is None:
                # Use data_root if available, otherwise use current directory
                yolo_output = args.data_root if args.data_root else '.'
            else:
                yolo_output = args.yolo_output_dir
            
            convert_to_yolo_pose(
                dataset=dataset,
                output_dir=yolo_output,
                data_root=args.data_root,
                img_prefix=args.img_prefix
            )
        
        print("\nDataset loading completed successfully!")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
