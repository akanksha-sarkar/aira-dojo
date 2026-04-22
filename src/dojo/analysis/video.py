import json
import ast
import re
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None
    import difflib


# =========================================================
# Load results
# =========================================================
def load_results(path):
    text = Path(path).read_text()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


# =========================================================
# Basic normalization
# =========================================================
def normalize_method_name(s: str) -> str:
    s = str(s).lower().strip()
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    s = s.replace("/", " ")
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def similarity(a: str, b: str) -> float:
    if fuzz is not None:
        return fuzz.token_sort_ratio(a, b)
    return 100 * difflib.SequenceMatcher(None, a, b).ratio()


# =========================================================
# Fuzzy clustering
# =========================================================
def build_fuzzy_method_map(all_methods, threshold=88):
    """
    Returns:
        raw_to_canonical: raw method -> fuzzy canonical method
        clusters: canonical method -> list of normalized variants
    """
    normalized_methods = [normalize_method_name(m) for m in all_methods]
    freq = Counter(normalized_methods)

    # most common names get priority as canonical labels
    sorted_methods = [m for m, _ in freq.most_common()]

    canonicals = []
    normalized_to_canonical = {}
    clusters = {}

    for method in sorted_methods:
        best_match = None
        best_score = -1

        for canon in canonicals:
            score = similarity(method, canon)
            if score > best_score:
                best_score = score
                best_match = canon

        if best_match is not None and best_score >= threshold:
            normalized_to_canonical[method] = best_match
            clusters[best_match].append(method)
        else:
            canonicals.append(method)
            normalized_to_canonical[method] = method
            clusters[method] = [method]

    raw_to_canonical = {}
    for raw in all_methods:
        norm = normalize_method_name(raw)
        raw_to_canonical[raw] = normalized_to_canonical[norm]

    return raw_to_canonical, clusters


def canonicalize_methods(methods, raw_to_canonical):
    return sorted(set(raw_to_canonical[m] for m in methods))


# =========================================================
# Helpers
# =========================================================


def sorted_runs(results_dict):
    filtered_results_dict = {}
    for run_id, run_data in results_dict.items():
        print(run_id)
        if type(run_data) != dict:
            continue
        filtered_results_dict[run_id] = run_data
    return filtered_results_dict.items()


def collect_all_methods(results_dict):
    all_methods = []
    for run_id, run_data in results_dict.items():
        if type(run_data) != dict:
            continue
        methods = run_data.get("program_analysis", {}).get("methods", [])
        all_methods.extend(methods)
    return all_methods


def print_clusters(clusters, min_size=2):
    print("\n===== Fuzzy Method Clusters =====")
    for canon, variants in sorted(clusters.items(), key=lambda x: (-len(x[1]), x[0])):
        if len(variants) >= min_size:
            print(f"{canon}: {variants}")


# =========================================================
# Animation
# =========================================================
def animate_method_occurrences(
    results_dict,
    threshold=88,
    top_k=12,
    interval=800,
    save_path=None,
    dpi=150,
    show_cluster_summary=True,
):
    """
    Animate cumulative method counts as programs are added one by one.

    Args:
        results_dict: nested experiment dict
        threshold: fuzzy-match threshold
        top_k: number of most frequent methods to show
        interval: ms between frames
        save_path: .gif or .mp4
        dpi: save resolution
    """
    all_methods = collect_all_methods(results_dict)
    raw_to_canonical, clusters = build_fuzzy_method_map(all_methods, threshold=threshold)

    if show_cluster_summary:
        print_clusters(clusters)

    runs = sorted_runs(results_dict)

    cumulative_counters = []
    frame_labels = []
    counter = Counter()

    for run_id, run_data in runs:
        if type(run_data) != dict:
            continue
        methods = run_data.get("program_analysis", {}).get("methods", [])
        methods = canonicalize_methods(methods, raw_to_canonical)

        counter.update(methods)
        cumulative_counters.append(counter.copy())

        acc = run_data.get("test_accuracy", None)
        if acc is not None:
            frame_labels.append(f"{run_id} | acc={acc:.3f}")
        else:
            frame_labels.append(run_id)

    fig, ax = plt.subplots(figsize=(10, 6))
    max_count = max(max(c.values()) for c in cumulative_counters) if cumulative_counters else 1

    def update(frame_idx):
        ax.clear()

        current_counts = cumulative_counters[frame_idx]
        top_items = current_counts.most_common(top_k)

        methods = [m for m, _ in top_items][::-1]
        counts = [c for _, c in top_items][::-1]

        ax.barh(methods, counts)
        ax.set_xlim(0, max_count + 1)
        ax.set_xlabel("Cumulative count")
        ax.set_ylabel("Method")
        ax.set_title(
            "Cumulative method occurrences through program iteration\n"
            f"Frame {frame_idx + 1}/{len(cumulative_counters)}: {frame_labels[frame_idx]}"
        )
        ax.grid(axis="x", alpha=0.25)

        for i, c in enumerate(counts):
            ax.text(c + 0.03, i, str(c), va="center")

    anim = FuncAnimation(
        fig,
        update,
        frames=len(cumulative_counters),
        interval=interval,
        repeat=False,
    )

    plt.tight_layout()

    if save_path is not None:
        if str(save_path).endswith(".gif"):
            anim.save(save_path, writer="pillow", dpi=dpi)
        else:
            anim.save(save_path, dpi=dpi)

    return anim, raw_to_canonical, clusters


# =========================================================
# Example usage
# =========================================================
if __name__ == "__main__":
    results = load_results("/home/eyl45/Sun/aira-dojo/src/dojo/analysis/runs/dtd/aSSL_backbone_unsupervised/k3/seed0/aira/2/analysis_dict.json")

    anim, raw_to_canonical, clusters = animate_method_occurrences(
        results,
        threshold=70,                 # lower if grouping is too strict
        top_k=40,
        interval=1000,
        save_path="viz_outputs/dtd/unsupervised/k3/method_occurrences.gif",
        show_cluster_summary=True,
    )

    print("\n===== Raw -> Canonical Mapping =====")
    for raw, canon in sorted(raw_to_canonical.items()):
        print(f"{raw} -> {canon}")

    plt.show()