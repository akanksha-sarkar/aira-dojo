import json
import ast
import re
import textwrap
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

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
    normalized_methods = [normalize_method_name(m) for m in all_methods]
    freq = Counter(normalized_methods)

    # Most common names become canonical first
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
def extract_step_num(run_id):
    """
    Example:
        0004_7694c9d8 -> 4
    """
    try:
        return int(str(run_id).split("_")[0])
    except Exception:
        return 10**9


def sorted_runs(results_dict):
    filtered = []
    for run_id, run_data in results_dict.items():
        if isinstance(run_data, dict):
            filtered.append((run_id, run_data))
    filtered.sort(key=lambda x: extract_step_num(x[0]))
    return filtered


def collect_all_methods(results_dict):
    all_methods = []
    for _, run_data in results_dict.items():
        if not isinstance(run_data, dict):
            continue
        methods = run_data.get("program_analysis", {}).get("methods", [])
        all_methods.extend(methods)
    return all_methods


def print_clusters(clusters, min_size=2):
    print("\n===== Fuzzy Method Clusters =====")
    for canon, variants in sorted(clusters.items(), key=lambda x: (-len(x[1]), x[0])):
        if len(variants) >= min_size:
            print(f"{canon}: {variants}")


def wrap_label(label, width=24):
    return "\n".join(textwrap.wrap(label, width=width))


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
    label_wrap_width=24,
):
    """
    Animate cumulative method counts as programs are added one by one.

    Improvements:
    - better margins so labels are not cut off
    - highlights methods appearing in the current step
    - cleaner visual styling
    """
    all_methods = collect_all_methods(results_dict)
    raw_to_canonical, clusters = build_fuzzy_method_map(all_methods, threshold=threshold)

    if show_cluster_summary:
        print_clusters(clusters)

    runs = sorted_runs(results_dict)

    cumulative_counters = []
    frame_labels = []
    frame_current_methods = []
    counter = Counter()

    for run_id, run_data in runs:
        methods = run_data.get("program_analysis", {}).get("methods", [])
        methods = canonicalize_methods(methods, raw_to_canonical)

        # Store which methods were used in this step
        current_step_methods = set(methods)
        frame_current_methods.append(current_step_methods)

        # Update cumulative counts
        counter.update(methods)
        cumulative_counters.append(counter.copy())

        acc = run_data.get("test_accuracy", None)
        if acc is not None:
            frame_labels.append(f"{run_id} | acc={acc:.3f}")
        else:
            frame_labels.append(str(run_id))

    if not cumulative_counters:
        raise ValueError("No valid runs found to animate.")

    max_count = max(max(c.values()) for c in cumulative_counters)

    # Bigger figure + more left margin
    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    fig.patch.set_facecolor("white")

    def update(frame_idx):
        ax.clear()
        ax.set_facecolor("white")

        current_counts = cumulative_counters[frame_idx]
        current_step_methods = frame_current_methods[frame_idx]

        top_items = current_counts.most_common(top_k)

        # reverse for horizontal bar chart top-down ordering
        methods = [m for m, _ in top_items][::-1]
        counts = [c for _, c in top_items][::-1]

        wrapped_methods = [wrap_label(m, width=label_wrap_width) for m in methods]

        # Highlight methods used in the current step
        colors = []
        edgecolors = []
        linewidths = []
        for m in methods:
            if m in current_step_methods:
                colors.append("tab:orange")
                edgecolors.append("black")
                linewidths.append(1.0)
            else:
                colors.append("lightgray")
                edgecolors.append("gray")
                linewidths.append(0.6)

        bars = ax.barh(
            wrapped_methods,
            counts,
            color=colors,
            edgecolor=edgecolors,
            linewidth=linewidths,
            height=0.72,
        )

        # X range with some room for labels
        ax.set_xlim(0, max_count + 1.5)

        # Styling
        ax.set_xlabel("Cumulative count", fontsize=12)
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=10, pad=10)
        ax.tick_params(axis="x", labelsize=10)

        # Clean spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Light grid
        ax.grid(axis="x", linestyle="--", alpha=0.25)

        # Main title
        ax.set_title(
            "Cumulative Method Occurrences Across Program Iterations",
            fontsize=15,
            pad=18,
            weight="bold",
        )

        # Subtitle-like text
        subtitle = f"Frame {frame_idx + 1}/{len(cumulative_counters)}   •   {frame_labels[frame_idx]}"
        ax.text(
            0.0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
        )

        # Add count labels at bar ends
        for bar, c in zip(bars, counts):
            ax.text(
                bar.get_width() + 0.08,
                bar.get_y() + bar.get_height() / 2,
                str(c),
                va="center",
                ha="left",
                fontsize=10,
            )

        # Show current-step highlighted methods in footer
        if current_step_methods:
            footer_methods = ", ".join(sorted(current_step_methods))
            footer_methods = textwrap.shorten(footer_methods, width=120, placeholder=" ...")
            ax.text(
                0.0,
                -0.16,
                f"Highlighted = methods used in this step: {footer_methods}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                color="tab:orange",
            )

    anim = FuncAnimation(
        fig,
        update,
        frames=len(cumulative_counters),
        interval=interval,
        repeat=False,
    )

    # Much better control than just tight_layout()
    fig.subplots_adjust(left=0.35, right=0.97, top=0.88, bottom=0.22)

    if save_path is not None:
        save_path = str(save_path)
        if save_path.endswith(".gif"):
            anim.save(save_path, writer=PillowWriter(fps=max(1, int(1000 / interval))), dpi=dpi)
        else:
            anim.save(save_path, dpi=dpi)

    return anim, raw_to_canonical, clusters


# =========================================================
# Example usage
# =========================================================
if __name__ == "__main__":
    results = load_results(
        "/home/eyl45/Sun/aira-dojo/src/dojo/analysis/runs/dtd/aSSL_backbone_unsupervised/k3/seed0/aira/2/analysis_dict.json"
    )

    anim, raw_to_canonical, clusters = animate_method_occurrences(
        results,
        threshold=70,
        top_k=20,   # 40 gets crowded; 15–25 usually looks better
        interval=1000,
        save_path="viz_outputs/dtd/unsupervised/k3/method_occurrences.gif",
        show_cluster_summary=True,
        label_wrap_width=24,
    )

    print("\n===== Raw -> Canonical Mapping =====")
    for raw, canon in sorted(raw_to_canonical.items()):
        print(f"{raw} -> {canon}")

    plt.show()