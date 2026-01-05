"""
Evaluator for cell segmentation programs.
"""
import sys
import os
# Add the Apptainer bind mount path to Python's module search
sys.path.insert(0, "/work")

print("✅ Added /work to sys.path")
print("Working dir:", os.getcwd())
import time
import logging
import importlib.util
from ctc_metrics.scripts.evaluate import evaluate_sequence
import shutil
from src.util import clear_dir, rename_sequence
from src.prepare import prepare 
from pathlib import Path
from src.dataset import CellSegDataset
import sys
import tifffile as tiff
import numpy as np

logging.basicConfig(
    level=logging.INFO, 
    format="[%(levelname)s] %(message)s",
    stream=sys.stdout, 
    force=True  
)

def write_pred_masks(seg_mask_list, val_ann_dir, out_dir, exts=(".tif", ".tiff")):
    """
    Write predicted instance masks to disk for CTC evaluation.

    Args:
        seg_mask_list (List[np.ndarray]): predicted instance-id masks (H x W), ideally uint16
        val_ann_dir (str or Path): directory containing GT masks (used only for filenames)
        out_dir (str or Path): directory to write predicted masks into
        exts (tuple): allowed GT extensions to match
    """
    val_ann_dir = Path(val_ann_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_mask_paths = sorted([p for p in val_ann_dir.iterdir() if p.suffix.lower() in exts])

    if len(gt_mask_paths) == 0:
        raise FileNotFoundError(f"No GT mask files found in {val_ann_dir} with extensions {exts}")

    if len(seg_mask_list) != len(gt_mask_paths):
        raise ValueError(
            f"Mismatch: {len(seg_mask_list)} preds vs {len(gt_mask_paths)} GT masks in {val_ann_dir}"
        )

    for pred_mask, gt_path in zip(seg_mask_list, gt_mask_paths):
        pred_mask = np.asarray(pred_mask)

        if pred_mask.dtype != np.uint16:
            pred_mask = pred_mask.astype(np.uint16, copy=False)

        out_path = out_dir / gt_path.name  
        tiff.imwrite(str(out_path), pred_mask)


def resolve_seq_paths(root_dir: str, seq: str):
    root_dir = str(root_dir)
    candidates = {
        "train_data": [os.path.join(root_dir, "data", "train", seq, "img")],
        "train_ann":  [os.path.join(root_dir, "data", "train", seq, "seg")],
        "val_data":   [os.path.join(root_dir, "data", "val",   seq, "img"),
                       os.path.join(root_dir, "val", seq, "img")],
        "val_ann":    [os.path.join(root_dir, "data", "val",   seq, "GT", "SEG"),
                       os.path.join(root_dir, "val", seq, "GT", "SEG")],
        "val_gt":     [os.path.join(root_dir, "data", "val",   seq, "GT"),
                       os.path.join(root_dir, "val", seq, "GT")],
    }

    def pick(lst):
        return next((p for p in lst if Path(p).exists()), lst[0])

    paths = {k: pick(v) for k, v in candidates.items()}
    return paths



def evaluate(program_path, root_dir, seed=42, seqs=("01", "02")):
    """
    Evaluate over one or more sequences and aggregate metrics.
    The explicit train_* / val_* args are kept for backward compatibility,
    but when seqs has multiple entries we derive per-seq paths from ROOT_DIR.
    """
    start_time = time.time()

    logging.info(f"Loading program from {program_path}")
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    if not hasattr(program, "cell_segmentation"):
        raise AttributeError("❌ The program must define a 'cell_segmentation' function.")

    root_dir = os.environ.get("ROOT_DIR", "/work")

    cache_dir = Path("/cache")
    if cache_dir.exists() and os.access(cache_dir, os.W_OK):
        base_pred_dir = cache_dir / "ctc_preds"
    else:
        scratch_dir = Path("/scratch")
        if scratch_dir.exists():
            job_id = os.environ.get("SLURM_JOBID", "generic")
            base_pred_dir = scratch_dir / str(job_id) / "ctc_preds"
        else:
            base_pred_dir = Path(".") / "tmp" / "ctc_preds"
    base_pred_dir.mkdir(parents=True, exist_ok=True)

    per_seq = {}
    seg_scores = []

    for seq in seqs:
        paths = resolve_seq_paths(root_dir, seq)

        t_data, t_ann = paths["train_data"], paths["train_ann"]
        v_data, v_ann = paths["val_data"], paths["val_ann"]
        gt_root = paths["val_gt"]

        logging.info(f"[SEQ {seq}] train_data={t_data}")
        logging.info(f"[SEQ {seq}] train_ann ={t_ann}")
        logging.info(f"[SEQ {seq}] val_data  ={v_data}")
        logging.info(f"[SEQ {seq}] val_ann   ={v_ann}")
        logging.info(f"[SEQ {seq}] gt_root   ={gt_root}")

        train_dataset = CellSegDataset(t_data, t_ann, split="train")
        val_dataset = CellSegDataset(v_data, v_ann, split="val")

        seg_mask_list = program.cell_segmentation(train_dataset, val_dataset)

        pred_dir = base_pred_dir / seq
        pred_dir.mkdir(parents=True, exist_ok=True)
        clear_dir(str(pred_dir)) 

        write_pred_masks(seg_mask_list, v_ann, pred_dir)

        metrics_seq = evaluate_sequence(gt=str(gt_root), res=str(pred_dir), metrics=["SEG"])
        seg = metrics_seq.get("SEG", None)
        if seg is None:
            raise RuntimeError(f"[SEQ {seq}] evaluate_sequence did not return SEG metric: {metrics_seq}")

        per_seq[seq] = {"SEG": float(seg), "fitness": float(seg)}
        seg_scores.append(float(seg))

        logging.info(f"[SEQ {seq}] metrics={per_seq[seq]}")

    fitness = float(np.mean(seg_scores)) if seg_scores else None

    metrics = {
        "fitness": fitness,
        "SEG": fitness,          
        "per_seq": per_seq,
        "n_seqs": len(per_seq),
        "eval_time_sec": float(time.time() - start_time),
    }
    return metrics

# Dummy main function for testing
# if __name__ == "__main__":
#     program_path = "/home/as2637/weco-cli/cell_seg/prog.py"
#     train_data_dir = "/share/j_sun/as2637/sciduc/cell_seg/k50/data/01/img"
#     train_ann_dir = "/share/j_sun/as2637/sciduc/cell_seg/k50/data/01/seg"
#     val_data_dir = "/share/j_sun/as2637/sciduc/cell_seg/data/val/01/img"
#     val_ann_dir = "/share/j_sun/as2637/sciduc/cell_seg/data/val/01/GT/SEG"
    
#     metric = evaluate(program_path, train_data_dir, train_ann_dir, val_data_dir, val_ann_dir)
#     print(metric)

if __name__ == "__main__":
    # Default paths are Apptainer-friendly

    root_dir = os.environ.get("ROOT_DIR", "/work") # this is a folder with partition like k100    
    program_path = os.path.join(root_dir, "program.py")
    print(f"  program_path: {program_path} (exists: {Path(program_path).exists()})")

    metrics = evaluate(program_path, root_dir, seqs=("01","02"), seed=42)
    print("METRICS:", metrics)
