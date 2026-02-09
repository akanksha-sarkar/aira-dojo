#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# User config
# -----------------------------
val_gt_src_01="/share/j_sun/as2637/BF-C2DL-HSC/01_GT/SEG"
val_imgs_src_01="/share/j_sun/as2637/BF-C2DL-HSC/01"

val_gt_src_02="/share/j_sun/as2637/BF-C2DL-HSC/02_GT/SEG"
val_imgs_src_02="/share/j_sun/as2637/BF-C2DL-HSC/02"

target_dir="/share/j_sun/as2637/sciduc/cell_seg/k5/data"

# -----------------------------
# Helpers
# -----------------------------
make_seq() {
  local seq_id="$1"         # "01" or "02"
  local gt_src="$2"         # .../XX_GT/SEG
  local img_src="$3"        # .../XX
  local out_base="$4"       # .../data

  local out_gt_seg="${out_base}/val/${seq_id}/GT/SEG"
  local out_gt_tra="${out_base}/val/${seq_id}/GT/TRA"
  local out_img="${out_base}/val/${seq_id}/img"

  mkdir -p "${out_gt_seg}" "${out_gt_tra}" "${out_img}"

  # Collect masks (sorted for deterministic renaming)
  mapfile -t masks < <(find "${gt_src}" -maxdepth 1 -type f -name "*.tif" | sort)
  if [[ ${#masks[@]} -eq 0 ]]; then
    echo "[ERROR] No .tif masks found in: ${gt_src}" >&2
    exit 1
  fi

  echo "[INFO] Seq ${seq_id}: found ${#masks[@]} masks"

  local new_idx=0
  for mask_path in "${masks[@]}"; do
    local mask_file frame img_file img_path new_mask_name new_img_name
    mask_file="$(basename "${mask_path}")"

    # Expect pattern like: man_seg0058.tif
    frame="$(echo "${mask_file}" | sed -nE 's/.*seg([0-9]+)\.tif/\1/p')"
    if [[ -z "${frame}" ]]; then
      echo "[WARN] Skipping unrecognized mask filename (expected *seg####.tif): ${mask_file}" >&2
      continue
    fi

    # Corresponding image: t####.tif
    img_file="t${frame}.tif"
    img_path="${img_src}/${img_file}"
    if [[ ! -f "${img_path}" ]]; then
      echo "[ERROR] Missing image for mask ${mask_file}: expected ${img_path}" >&2
      exit 1
    fi

    new_mask_name="$(printf "man_seg%04d.tif" "${new_idx}")"
    new_img_name="$(printf "t%04d.tif" "${new_idx}")"

    # Copy to SEG + TRA (same files), and the corresponding image
    cp -f "${mask_path}" "${out_gt_seg}/${new_mask_name}"
    cp -f "${mask_path}" "${out_gt_tra}/${new_mask_name}"
    cp -f "${img_path}"  "${out_img}/${new_img_name}"

    new_idx=$((new_idx + 1))
  done

  echo "[INFO] Seq ${seq_id}: wrote ${new_idx} (mask,image) pairs to:"
  echo "       ${out_gt_seg}"
  echo "       ${out_gt_tra}"
  echo "       ${out_img}"
}

# -----------------------------
# Main
# -----------------------------
mkdir -p "${target_dir}/val"

make_seq "01" "${val_gt_src_01}" "${val_imgs_src_01}" "${target_dir}"
make_seq "02" "${val_gt_src_02}" "${val_imgs_src_02}" "${target_dir}"

echo "Validation dataset created successfully"