"""
Script to preapre different data efficiency divides. 
"""
# Data_dir to be the SciDuc/cell_seg directory 
# BF-C2DL-HSC
import os
import re
import glob
import math
import random
from pathlib import Path
import argparse
from typing import Optional, Tuple

import os, re
from pathlib import Path
from typing import Optional, Tuple, List

def rename_val_sequences(val_root: str,
                         img_dirname: str = "img",
                         seg_dir: Tuple[str, str] = ("GT", "SEG"),
                         tra_dir: Tuple[str, str] = ("GT", "TRA"),
                         pad: int = 4,
                         img_prefix: str = "t",
                         seg_prefix: str = "man_seg",
                         img_ext: str = ".tif",
                         seg_ext: str = ".tif",
                         dry_run: bool = False):
    """
    Under val_root/<seq>/(img, GT/SEG, GT/TRA):
      img -> t0000.tif, t0001.tif, ...
      GT/SEG -> man_seg0000.tif, man_seg0001.tif, ...
      GT/TRA -> rebuilt as symlinks to GT/SEG after rename.
    """
    val_root = Path(val_root)
    seq_dirs = sorted([p for p in val_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not seq_dirs:
        print(f"[rename] no sequences under {val_root}"); return

    def idx(p: Path) -> Optional[int]:
        m = re.findall(r"(\d+)", p.stem)
        return int(m[-1]) if m else None

    def two_phase_rename(pairs: List[Tuple[Path, Path]]):
        # move src -> tmp, then tmp -> dst to avoid collisions
        temps: List[Tuple[Path, Path]] = []
        for s, d in pairs:
            t = s.with_name(f".__tmp__{s.name}")
            # ensure uniqueness
            i = 0
            while t.exists():
                i += 1
                t = s.with_name(f".__tmp__{i}__{s.name}")
            s.rename(t)
            temps.append((t, d))
        for t, d in temps:
            if d.exists() or d.is_symlink():
                d.unlink()
            t.rename(d)

    for seq in seq_dirs:
        img_dir = seq / img_dirname
        seg_path = seq.joinpath(*seg_dir)   # GT/SEG
        tra_path = seq.joinpath(*tra_dir)   # GT/TRA

        if not (img_dir.is_dir() and seg_path.is_dir()):
            print(f"[warn] {seq.name}: missing {img_dirname}/ or {'/'.join(seg_dir)}/; skipping"); continue
        tra_path.mkdir(parents=True, exist_ok=True)

        imgs = sorted([p for p in img_dir.iterdir() if p.is_file()])
        segs = sorted([p for p in seg_path.iterdir() if p.is_file()])

        img_by = {i: p for p in imgs if (i := idx(p)) is not None}
        seg_by = {i: p for p in segs if (i := idx(p)) is not None}
        common = sorted(set(img_by) & set(seg_by))
        if not common:
            print(f"[warn] {seq.name}: no common indices; skipping"); continue

        img_pairs = []
        seg_pairs = []
        for new_i, old_i in enumerate(common):
            img_pairs.append((img_by[old_i], img_dir / f"{img_prefix}{new_i:0{pad}d}{img_ext}"))
            seg_pairs.append((seg_by[old_i], seg_path / f"{seg_prefix}{new_i:0{pad}d}{seg_ext}"))

        # manifest (brief)
        with (seq / "rename_manifest.txt").open("w") as f:
            f.write(f"# {seq.name}: {len(common)} pairs\n")
            for (si, di), (sm, dm) in zip(img_pairs, seg_pairs):
                f.write(f"{si.name} -> {di.name} | {sm.name} -> {dm.name}\n")

        if dry_run:
            print(f"[dry-run] {seq.name}: would rename {len(common)} pairs"); continue

        two_phase_rename(img_pairs)
        two_phase_rename(seg_pairs)

        # Rebuild TRA as links to SEG (mirror)
        for d in tra_path.iterdir():
            if d.is_file() or d.is_symlink():
                d.unlink()
        for seg_file in sorted(seg_path.iterdir()):
            dst = tra_path / seg_file.name
            # make TRA symlink point to SEG relative path
            rel = os.path.relpath(seg_file, start=tra_path)
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(rel, dst)

        print(f"[rename] {seq.name}: renamed {len(common)} pairs and mirrored GT/SEG -> GT/TRA")


def prepare_validation(data_dir: str, out_dir: str):
    """
    Prepare validation data by symlinking, per sequence, the images that have GT masks,
    and placing GT in nested folders:
        <seq>/GT/SEG/  (symlinks to masks)
        <seq>/GT/TRA/  (symlinks to SEG entries; i.e., a mirror of SEG)
        <seq>/img/     (symlinks to matched images)
    """
    data_dir = Path(data_dir)
    out_root = Path(out_dir) / "val"
    out_root.mkdir(parents=True, exist_ok=True)

    def find_img_dir(seq_dir: Path) -> Path | None:
        # Prefer "<seq>_img", "<seq>_IMG", then "img", "IMG", then any "*_img" or "*_IMG"
        seq_name = seq_dir.name
        candidates = [
            seq_dir / f"{seq_name}_img",
            seq_dir / f"{seq_name}_IMG",
            seq_dir / f"{seq_name}_Img",
            seq_dir / "img",
            seq_dir / "IMG",
            seq_dir / "Img",
        ]
        # Also check for any directories ending with _img or _IMG (case-insensitive)
        for p in seq_dir.iterdir():
            if p.is_dir() and p.name.lower().endswith("_img"):
                candidates.append(p)
        
        for p in candidates:
            if p.is_dir():
                return p
        return None

    def find_gt_dir(seq_dir: Path) -> Path | None:
        # Common GT layouts (order matters)
        prefs = [
            seq_dir / f"{seq_dir.name}_GT" / "SEG",
            seq_dir / f"{seq_dir.name}_GT",
            seq_dir / "GT" / "SEG",
            seq_dir / "GT",
            seq_dir / "err_seg" / "SEG",
            seq_dir / "err_seg",
        ]
        for p in prefs:
            if p.is_dir():
                return p
        # Fallback: any folder containing man_seg*.tif
        for p in seq_dir.rglob("*"):
            if p.is_dir() and list(p.glob("man_seg*.tif")):
                return p
        return None

    def idx_from_name(path: Path):
        m = re.findall(r"(\d+)", path.stem)
        return int(m[-1]) if m else None

    def symlink_force(src: Path, dst: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
        except FileNotFoundError:
            pass
        os.symlink(src, dst)

    # Iterate sequences
    seq_dirs = [p for p in data_dir.iterdir() if p.is_dir()]
    if not seq_dirs:
        raise FileNotFoundError(f"No sequences found in {data_dir}")

    for seq_dir in sorted(seq_dirs, key=lambda p: p.name):
        img_dir = find_img_dir(seq_dir)
        gt_dir  = find_gt_dir(seq_dir)

        if img_dir is None:
            print(f"[warn] No image dir found in {seq_dir}; skipping")
            continue
        if gt_dir is None:
            print(f"[warn] No GT dir found in {seq_dir}; skipping")
            continue

        # Collect images
        img_files = sorted(
            [Path(p) for ext in ("*.tif", "*.tiff", "*.png", "*.jpg")
             for p in glob.glob(str(img_dir / ext))]
        )
        if not img_files:
            print(f"[warn] No images in {img_dir}; skipping")
            continue

        # Collect GT masks
        mask_files = sorted(
            [Path(p) for ext in ("man_seg*.tif", "man_seg*.tiff", "*.tif", "*.tiff", "*.png")
             for p in glob.glob(str(gt_dir / ext))]
        )
        if not mask_files:
            print(f"[warn] No GT masks in {gt_dir}; skipping {seq_dir.name}")
            continue

        # Index images by numeric suffix
        idx_to_img = {}
        for p in img_files:
            idx = idx_from_name(p)
            if idx is not None and idx not in idx_to_img:
                idx_to_img[idx] = p

        # Prepare output dirs
        out_seq     = out_root / seq_dir.name
        out_img     = out_seq / "img"
        out_gt_seg  = out_seq / "GT" / "SEG"
        out_gt_tra  = out_seq / "GT" / "TRA"
        out_img.mkdir(parents=True, exist_ok=True)
        out_gt_seg.mkdir(parents=True, exist_ok=True)
        out_gt_tra.mkdir(parents=True, exist_ok=True)

        # Link matched images + masks (into SEG)
        linked = 0
        missing_img = []
        manifest_lines = []
        seg_links_created = []  # (seg_dst_path)
        for m in mask_files:
            idx = idx_from_name(m)
            if idx is None:
                continue
            src_img = idx_to_img.get(idx)
            if src_img is None:
                missing_img.append((idx, m.name))
                continue

            # image link
            img_dst = out_img / src_img.name
            symlink_force(src_img, img_dst)

            # gt/SEG link (keep original mask filename)
            seg_dst = out_gt_seg / m.name
            symlink_force(m, seg_dst)
            seg_links_created.append(seg_dst)

            linked += 1
            manifest_lines.append(f"{m.name} -> {src_img.name}")

        # Mirror SEG into TRA (as symlinks pointing to SEG entries)
        for seg_dst in seg_links_created:
            tra_dst = out_gt_tra / seg_dst.name
            # Link TRA file to the SEG symlink (so it's literally a copy of SEG)
            rel_target = os.path.relpath(seg_dst, start=tra_dst.parent)
            try:
                if tra_dst.exists() or tra_dst.is_symlink():
                    tra_dst.unlink()
            except FileNotFoundError:
                pass
            os.symlink(rel_target, tra_dst)

        # Manifest
        with (out_seq / "manifest.txt").open("w") as f:
            f.write(f"# seq: {seq_dir.name}\n")
            f.write(f"# img_dir: {img_dir}\n")
            f.write(f"# gt_src : {gt_dir}\n")
            f.write(f"# linked : {linked}\n")
            if missing_img:
                f.write(f"# masks with no matching image: {len(missing_img)}\n")
            f.write("#\n# mask -> image (SEG mirrored to TRA)\n")
            for line in manifest_lines:
                f.write(line + "\n")
            if missing_img:
                f.write("\n# Missing image for these masks:\n")
                for idx, name in missing_img:
                    f.write(f"# idx={idx:04d} mask={name}\n")

        print(f"[val] {seq_dir.name}: linked {linked} pairs; GT/SEG & GT/TRA prepared at {out_seq/'GT'}")


def prepare(split_ratio: float, data_dir: str, out_dir: str, seed: int = 42):
    """
    Create a k_{split_ratio*100} subset by symlinking a fraction of frames
    from each sequence into: out_dir/k_xx/<seq>/{img,err_seg}/

    Expected input layout (per sequence folder 'seq'):
        data_dir/seq/img/*.tif
        data_dir/seq/err_seg/            (either *.tif here or a SEG/ subfolder)

    Output layout:
        out_dir/k_xx/seq/img/      -> symlinks to selected images
        out_dir/k_xx/seq/err_seg/  -> symlinks to corresponding masks

    Notes:
      - Selection is on a per-sequence basis.
      - Matching is by the last integer index found in filenames
        (e.g., t0007.tif ↔ man_seg0007.tif / mask0007.tif).
    """
    if not (0.0 <= split_ratio <= 1.0):
        raise ValueError("split_ratio must be in [0.0, 1.0]")

    data_dir = Path(data_dir)
    out_dir  = Path(out_dir)
    root_out = out_dir / "data" 
    root_out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    def _idx_from_name(p: Path) -> int | None:
        m = re.findall(r"(\d+)", p.stem)
        return int(m[-1]) if m else None

    # helper: safe symlink (overwrite if exists)
    def _symlink(src: Path, dst: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
        except FileNotFoundError:
            pass
        os.symlink(src, dst)

    # helper:  case matching
    def _find_img_dir(seq_path: Path, seq_name: str) -> Path | None:
        """Try to find image directory with various case combinations."""
        candidates = [
            seq_path / f"{seq_name}_img",
            seq_path / f"{seq_name}_IMG",
            seq_path / f"{seq_name}_Img",
            seq_path / "img",
            seq_path / "IMG",
        ]
        for cand in candidates:
            if cand.is_dir():
                return cand
        return None
    
    # helper: find err_seg directory with flexible case matching
    def _find_err_seg_dir(seq_path: Path, seq_name: str) -> Path | None:
        """Try to find error segmentation directory with various case combinations."""
        candidates = [
            seq_path / f"{seq_name}_ERR_SEG",
            seq_path / f"{seq_name}_err_seg",
            seq_path / f"{seq_name}_Err_Seg",
            seq_path / "err_seg",
            seq_path / "ERR_SEG",
            seq_path / "Err_Seg",
        ]
        for cand in candidates:
            if cand.is_dir():
                return cand
        return None

    # collect sequences: any directory under data_dir that contains an img folder
    seq_dirs = []
    for p in data_dir.iterdir():
        if not p.is_dir():
            continue
        # Try to find any img directory (case-insensitive check)
        img_candidates = [
            p / f"{p.name}_img",
            p / f"{p.name}_IMG",
            p / f"{p.name}_Img",
            p / "img",
            p / "IMG",
        ]
        if any(cand.is_dir() for cand in img_candidates):
            seq_dirs.append(p)
    
    if not seq_dirs:
        raise FileNotFoundError(f"No sequences with an img folder found in {data_dir}")

    for seq_path in sorted(seq_dirs):
        seq = seq_path.name
        img_dir = _find_img_dir(seq_path, seq)
        err_seg_dir = _find_err_seg_dir(seq_path, seq)
        
        if img_dir is None:
            print(f"[warn] No image directory found in {seq_path}; skipping sequence '{seq}'")
            continue
        if err_seg_dir is None:
            print(f"[warn] No error segmentation directory found in {seq_path}; skipping sequence '{seq}'")
            continue


        # list images (tif/tiff/png)
        img_files = sorted(
            [Path(p) for ext in ("*.tif", "*.tiff", "*.png", "*.jpg")
             for p in glob.glob(str(img_dir / ext))]
        )
        if not img_files:
            print(f"[warn] No images found in {img_dir}; skipping sequence '{seq}'")
            continue

        # choose how many to keep
        if split_ratio == 0.0:
            keep_imgs = []
        elif split_ratio == 1.0:
            keep_imgs = img_files
        else:
            n_total = len(img_files)
            n_keep  = max(1, int(round(n_total * split_ratio)))
            # deterministic shuffle, then take first n_keep
            shuffled = img_files[:]
            rng.shuffle(shuffled)
            keep_imgs = sorted(shuffled[:n_keep], key=lambda p: p.name) 

        # build index set from kept images
        keep_indices = {}
        for p in keep_imgs:
            idx = _idx_from_name(p)
            if idx is not None:
                keep_indices[idx] = p
            else:
                # fallback: keep exact name match later
                keep_indices[p.stem] = p 

        # find all candidate masks
        mask_files = sorted(
            [Path(p) for ext in ("*.tif", "*.tiff", "*.png")
             for p in glob.glob(str(err_seg_dir / ext))]
        )
        # map masks by extracted index
        mask_by_idx: dict[int | str, Path] = {}
        for m in mask_files:
            mi = _idx_from_name(m)
            key = mi if mi is not None else m.stem
            mask_by_idx[key] = m

        # prepare output dirs
        seq_out_img     = root_out / seq / f"{seq}_img"
        seq_out_err_seg = root_out / seq / f"{seq}_ERR_SEG"
        seq_out_img.mkdir(parents=True, exist_ok=True)
        seq_out_err_seg.mkdir(parents=True, exist_ok=True)

        # symlink selected images
        for p in keep_imgs:
            dst = seq_out_img / p.name
            _symlink(p, dst)

        # symlink corresponding masks
        linked_masks = 0
        for key, p_img in keep_indices.items():
            m = mask_by_idx.get(key, None)
            if m is None:
                # Try tolerant match: exact four-digit zero-padded if key is int
                if isinstance(key, int):
                    patt = re.compile(rf".*[^0-9]*{key:04d}\b")
                    candidates = [mf for mf in mask_files if patt.match(mf.stem)]
                    if candidates:
                        m = candidates[0]
            if m is not None:
                _symlink(m, seq_out_err_seg / m.name)

        # info for reproducibility
        manifest = root_out / seq / "info.txt"
        with manifest.open("w") as f:
            f.write(f"# sequence: {seq}\n")
            f.write(f"# split_ratio: {split_ratio}\n")
            f.write(f"# total_images: {len(img_files)}  kept: {len(keep_imgs)}\n")
            f.write("# kept frames:\n")
            for p in keep_imgs:
                f.write(p.name + "\n")

        print(f"[ok] {seq}: kept {len(keep_imgs)}/{len(img_files)} frames "
              f"→ {seq_out_img} and {seq_out_err_seg}")


#def main():
#     parser = argparse.ArgumentParser(
#         description="Create data-efficiency subsets by symlinking a fraction of each sequence."
#     )

#     parser.add_argument(
#         "--split-ratio", "-r",
#         type=float,
#         default=0.5,
#         help="Ratio of frames to keep (between 0.0 and 1.0). Default: 0.5"
#     )

#     parser.add_argument(
#         "--data-dir", "-d",
#         type=str,
#         default="/share/j_sun/as2637/cell_seg/BF-C2DL-HSC",
#         help="Path to the dataset directory containing sequences (each with img/ and err_seg/)."
#     )

#     parser.add_argument(
#         "--out-dir", "-o",
#         type=str,
#         default="/home/as2637/segment",
#         help="Path to output directory where subset folders (k_xx) will be created."
#     )

#     parser.add_argument(
#         "--seed", "-s",
#         type=int,
#         default=42,
#         help="Random seed for reproducible sampling. Default: 42"
#     )

#     args = parser.parse_args()

#     # Run the function
#     #prepare(
#     #    split_ratio=args.split_ratio,
#     #    data_dir=args.data_dir,
#     #    out_dir=args.out_dir,
#     #    seed=args.seed
#     #)
# prepare_validation(
#             data_dir="/share/j_sun/as2637/sciduc/cell_seg/data/train",
#             out_dir="/share/j_sun/as2637/sciduc/cell_seg/data"
#         )
#    rename_val_sequences(val_root="/share/j_sun/as2637/sciduc/cell_seg/data/val")
# if __name__ == "__main__":
#     main()