#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# User config
# -----------------------------
ratio="${1:-0.5}"          # allow: ./make_partition.sh 0.25
seed="${2:-42}"            # allow: ./make_partition.sh 0.25 123

# TARGET DIRECTORY
train_data_base_dir="/share/j_sun/as2637/sciduc/cell_seg"

# Sequence roots (images)
train_img_01="/share/j_sun/as2637/BF-C2DL-HSC/01"
train_img_02="/share/j_sun/as2637/BF-C2DL-HSC/02"

# Training GT segmentations (CTC-style "silver truth")
train_gt_seg_01="/share/j_sun/as2637/BF-C2DL-HSC/01_ST/SEG"
train_gt_seg_02="/share/j_sun/as2637/BF-C2DL-HSC/02_ST/SEG"

# Validation GT segmentations (frames here must be excluded from training subset)
# Filenames: man_seg####.tiff (or .tif)
val_01="/share/j_sun/as2637/BF-C2DL-HSC/01_GT/SEG"
val_02="/share/j_sun/as2637/BF-C2DL-HSC/02_GT/SEG"

# Other files 
other_files_dir="/share/j_sun/as2637/cell_seg_backup/temp"

# Image / seg extensions
img_ext="tif"   # adjust if your images are .tiff
seg_glob="man_seg*.tif*"  # supports .tif and .tiff

# -----------------------------
# Derived output paths
# -----------------------------
k=$(awk -v r="$ratio" 'BEGIN { printf("%d", r*100 + 0.5) }')   # 0.5 -> 50
out_dir="${train_data_base_dir}/k${k}"
out_data="${out_dir}/data/train"

echo "[INFO] ratio=$ratio (k=${k}), seed=$seed"
echo "[INFO] output dir: ${out_dir}"

mkdir -p "${out_data}/01/img" "${out_data}/01/seg"
mkdir -p "${out_data}/02/img" "${out_data}/02/seg"

# -----------------------------
# Helper: sample + copy for one sequence
# -----------------------------
sample_sequence () {
  local seq="$1"
  local img_dir="$2"
  local st_seg_dir="$3"
  local gt_seg_dir="$4"

  local out_img="${out_data}/${seq}/img"
  local out_seg="${out_data}/${seq}/seg"

  echo "[INFO] --- Sequence ${seq} ---"
  echo "[INFO] img_dir=${img_dir}"
  echo "[INFO] st_seg_dir=${st_seg_dir}"
  echo "[INFO] gt_seg_dir=${gt_seg_dir}"

  #  Validation IDs from GT masks: man_seg####.tif(f) -> ####
  local tmp_val_ids
  tmp_val_ids="$(mktemp)"
  find "${gt_seg_dir}" -maxdepth 1 -type f -name "${seg_glob}" -printf "%f\n" \
    | sed -nE 's/^man_seg([0-9]{4}).*/\1/p' \
    | sort -u > "${tmp_val_ids}"

  local n_val
  n_val=$(wc -l < "${tmp_val_ids}" | tr -d ' ')
  echo "[INFO] found ${n_val} validation IDs to exclude"

  # Candidate training image filenames + extracted 4-digit ID
  local tmp_candidates
  tmp_candidates="$(mktemp)"

  find "${img_dir}" -maxdepth 1 -type f -name "*.${img_ext}" -printf "%f\n" \
    | awk '
      match($0, /[0-9]{4}/) { print $0 "\t" substr($0, RSTART, RLENGTH) }
    ' \
    | sort -u > "${tmp_candidates}.with_ids"

  #  Exclude any candidate whose ID is in validation set
  awk -F'\t' 'NR==FNR {val[$1]=1; next} !(($2) in val) {print $1}' \
    "${tmp_val_ids}" "${tmp_candidates}.with_ids" > "${tmp_candidates}"

  local n_total n_keep
  n_total=$(wc -l < "${tmp_candidates}" | tr -d ' ')
  if [[ "${n_total}" -eq 0 ]]; then
    echo "[WARN] No training candidates left after exclusion for seq ${seq}."
    rm -f "${tmp_val_ids}" "${tmp_candidates}" "${tmp_candidates}.with_ids"
    return 0
  fi

  n_keep=$(awk -v n="${n_total}" -v r="${ratio}" 'BEGIN { k=int(n*r + 0.5); if(k<1) k=1; print k }')
  echo "[INFO] candidates=${n_total}, keeping=${n_keep}"

  # Deterministic shuffle with seed and select n_keep
  local tmp_selected
  tmp_selected="$(mktemp)"
  python - <<PY > "${tmp_selected}"
import random
random.seed(int("${seed}"))
files = [ln.strip() for ln in open("${tmp_candidates}") if ln.strip()]
random.shuffle(files)
print("\n".join(files[:int("${n_keep}")]))
PY

  #  Copy selected images + matching ST seg masks by ID (man_seg####.tif*)
  local manifest="${out_dir}/manifest_${seq}.txt"
  : > "${manifest}"

  while IFS= read -r fname; do
    [[ -z "${fname}" ]] && continue

    # Extract first 4-digit ID from image filename
    id="$(echo "${fname}" | sed -nE 's/.*([0-9]{4}).*/\1/p' | head -n1)"
    if [[ -z "${id}" ]]; then
      echo "[WARN] could not extract 4-digit id from ${fname}, skipping"
      continue
    fi

    local src_img="${img_dir}/${fname}"

    # Find ST seg file matching id: man_seg####.(tif|tiff)
    local src_seg
    src_seg="$(ls -1 "${st_seg_dir}/man_seg${id}."tif* 2>/dev/null | head -n1 || true)"
    if [[ -z "${src_seg}" ]]; then
      echo "[WARN] missing ST seg for id=${id}: expected ${st_seg_dir}/man_seg${id}.tif*"
      continue
    fi

    cp --update=none "${src_img}" "${out_img}/"
    cp --update=none "${src_seg}" "${out_seg}/"
    echo "${fname}" >> "${manifest}"
  done < "${tmp_selected}"

  echo "[INFO] wrote manifest: ${manifest}"
  echo "[INFO] output imgs: $(ls -1 "${out_img}" | wc -l | tr -d ' ')"
  echo "[INFO] output segs: $(ls -1 "${out_seg}" | wc -l | tr -d ' ')"

  rm -f "${tmp_val_ids}" "${tmp_candidates}" "${tmp_candidates}.with_ids" "${tmp_selected}"
}

# -----------------------------
# Run for both sequences
# -----------------------------
sample_sequence "01" "${train_img_01}" "${train_gt_seg_01}" "${val_01}"
sample_sequence "02" "${train_img_02}" "${train_gt_seg_02}" "${val_02}"

# -----------------------------
# Copy other files into k{ratio} directory
# -----------------------------
if [[ -d "${other_files_dir}" ]]; then
  echo "[INFO] Copying other files from ${other_files_dir} into ${out_dir}"
  cp -a "${other_files_dir}/." "${out_dir}/"
else
  echo "[WARN] other_files_dir does not exist: ${other_files_dir}"
fi



echo "[DONE] Created partition at: ${out_dir}"
echo "[DONE] Layout: ${out_data}/{01,02}/{img,seg}"