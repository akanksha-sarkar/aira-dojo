import json
import ast
import re
from pathlib import Path
from collections import Counter
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None
    import difflib


# =========================================================
# Parsing
# =========================================================
def load_results(path):
    """
    Loads either:
      - valid JSON
      - or a pasted Python dict / almost-JSON blob
    """
    text = Path(path).read_text()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(text)
    except Exception as e:
        raise ValueError(f"Could not parse {path}: {e}")


# =========================================================
# Simple cost helpers
# =========================================================
def total_cost(prompt_tokens, completion_tokens,
               prompt_price_per_1m, completion_price_per_1m):
    """
    Returns total cost in dollars.
    """
    return (
        (prompt_tokens / 1_000_000) * prompt_price_per_1m
        + (completion_tokens / 1_000_000) * completion_price_per_1m
    )


def total_experiment_cost(results_dict,
                          prompt_price_per_1m,
                          completion_price_per_1m):
    total = 0.0
    for run in results_dict.values():
        if type(run) != dict:
            continue
        total += total_cost(
            run.get("prompt_tokens", 0),
            run.get("completion_tokens", 0),
            prompt_price_per_1m,
            completion_price_per_1m,
        )
    return total


# =========================================================
# Flatten nested results dict -> DataFrame
# =========================================================
def results_to_dataframe(results_dict,
                         prompt_price_per_1m=None,
                         completion_price_per_1m=None):
    rows = []

    for run_id, run_data in results_dict.items():
        if type(run_data) != dict:
            continue
        analysis = run_data.get("program_analysis", {})

        methods = analysis.get("methods", []) or []
        models = analysis.get("models", []) or []

        used_dino = bool(analysis.get("used_dinov2", False))
        used_clip = bool(analysis.get("used_clip", False))

        model_text = " ".join(models).lower()
        used_dino = used_dino or ("dino" in model_text)
        used_clip = used_clip or ("clip" in model_text)

        row = {
            "run_id": run_id,
            "program_name": run_data.get("program_name"),
            "eval_name": run_data.get("eval_name"),
            "test_accuracy": run_data.get("test_accuracy"),
            "models": models,
            "methods": methods,
            "n_methods": len(methods),
            "uses_training": analysis.get("uses training"),
            "summary": analysis.get("summary"),
            "used_dinov2": used_dino,
            "used_clip": used_clip,
            "prompt_tokens": run_data.get("prompt_tokens", 0),
            "completion_tokens": run_data.get("completion_tokens", 0),
            "total_tokens": run_data.get("total_tokens", 0),
        }

        if prompt_price_per_1m is not None and completion_price_per_1m is not None:
            row["llm_cost"] = total_cost(
                row["prompt_tokens"],
                row["completion_tokens"],
                prompt_price_per_1m,
                completion_price_per_1m,
            )
        else:
            row["llm_cost"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)

    def backbone_bucket(row):
        if row["used_dinov2"] and row["used_clip"]:
            return "Both"
        elif row["used_dinov2"]:
            return "DINOv2 only"
        elif row["used_clip"]:
            return "CLIP only"
        else:
            return "Neither / Other"

    df["backbone_bucket"] = df.apply(backbone_bucket, axis=1)
    return df


# =========================================================
# Method canonicalization
# =========================================================
def normalize_method_name(s: str) -> str:
    s = str(s).lower().strip()

    s = s.replace("_", " ")
    s = s.replace("-", " ")
    s = s.replace("/", " ")
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()

    replacements = {
        "semisupervised": "semi supervised",
        "pseudo labeling": "pseudo label",
        "pseudo labels": "pseudo label",
        "pseudolabeling": "pseudo label",
        "test time augmentation": "tta",
        "k nn": "knn",
        "nearest centroid": "centroid",
        "multi crop": "multicrop",
        "two view": "multi view",
        "view consistency": "consistency",
        "consistency objectives": "consistency",
        "consistency objective": "consistency",
        "feature extraction": "feature extract",
        "feature extractor": "feature extract",
    }

    for old, new in replacements.items():
        s = s.replace(old, new)

    s = re.sub(r"\s+", " ", s).strip()
    return s


def similarity(a: str, b: str) -> float:
    if fuzz is not None:
        return fuzz.token_sort_ratio(a, b)
    return 100 * difflib.SequenceMatcher(None, a, b).ratio()


def build_method_canonical_map(all_methods, threshold=88):
    normalized = [normalize_method_name(m) for m in all_methods]
    freq = Counter(normalized)
    sorted_methods = [m for m, _ in freq.most_common()]

    canonicals = []
    canonical_map = {}
    clusters = {}

    for method in sorted_methods:
        matched = None
        best_score = -1

        for canon in canonicals:
            score = similarity(method, canon)
            if score > best_score:
                best_score = score
                matched = canon

        if matched is not None and best_score >= threshold:
            canonical_map[method] = matched
            clusters[matched].append(method)
        else:
            canonicals.append(method)
            canonical_map[method] = method
            clusters[method] = [method]

    return canonical_map, clusters


def apply_manual_aliases(method_name: str) -> str:
    """
    Maps fuzzy-cleaned names into broader semantic buckets.
    Edit this table as needed after inspecting your clusters.
    """
    alias_map = {
        "pseudo label": "pseudo-labeling",
        "confidence gated pseudo label": "pseudo-labeling",

        "transductive gaussian em": "gaussian em",
        "transductive gaussian model": "gaussian em",
        "em algorithm": "gaussian em",
        "em style refinement": "gaussian em",
        "class conditional gaussian modeling": "gaussian em",
        "soft responsibilities": "gaussian em",
        "class prior regularization": "gaussian em",

        "pca": "pca / whitening",
        "pca whitening": "pca / whitening",
        "pca projection": "pca / whitening",

        "consistency": "consistency / multi-view",
        "two view consistency": "consistency / multi-view",
        "multi view feature extraction": "consistency / multi-view",
        "two view feature extraction": "consistency / multi-view",

        "tta": "tta / multi-crop",
        "multicrop averaging": "tta / multi-crop",
        "multi crop averaging": "tta / multi-crop",

        "label propagation": "label propagation / graph methods",
        "knn graph": "label propagation / graph methods",

        "prototype refinement": "prototype methods",

        "nearest centroid cosine calibration": "cosine calibration",
        "cosine calibration": "cosine calibration",

        "feature extract": "feature extraction",
        "data augmentation": "data augmentation",
        "semi supervised learning": "semi-supervised learning",
        "transductive semi supervised learning": "semi-supervised learning",
        "frozen backbone": "frozen backbone",
        "adamw optimizer": "adamw optimizer",
        "label smoothing": "label smoothing",
    }

    return alias_map.get(method_name, method_name)


def add_canonical_methods(df, threshold=88):
    df = df.copy()

    all_methods = []
    for methods in df["methods"]:
        all_methods.extend(methods)

    canonical_map, clusters = build_method_canonical_map(all_methods, threshold=threshold)

    raw_to_final = {}
    for raw_method in all_methods:
        norm = normalize_method_name(raw_method)
        fuzzy_canon = canonical_map[norm]
        final_canon = apply_manual_aliases(fuzzy_canon)
        raw_to_final[raw_method] = final_canon

    df["methods_canonical"] = df["methods"].apply(
        lambda methods: sorted(set(raw_to_final[m] for m in methods))
    )
    df["n_methods_canonical"] = df["methods_canonical"].apply(len)

    return df, raw_to_final, clusters


def print_method_clusters(clusters):
    print("\n===== Fuzzy Method Clusters =====")
    for canon, variants in sorted(clusters.items(), key=lambda x: (-len(x[1]), x[0])):
        if len(variants) > 1:
            print(f"{canon}: {variants}")


# =========================================================
# Plot helpers
# =========================================================
def _get_method_col(canonical=True):
    return "methods_canonical" if canonical else "methods"


def _get_n_method_col(canonical=True):
    return "n_methods_canonical" if canonical else "n_methods"


# =========================================================
# Plot 1: method frequency
# =========================================================
def plot_method_frequency(df, save_path=None, top_k=None, canonical=True):
    method_col = _get_method_col(canonical)

    method_counter = Counter()
    for methods in df[method_col]:
        method_counter.update(methods)

    method_df = pd.DataFrame(
        [{"method": k, "count": v} for k, v in method_counter.items()]
    ).sort_values("count", ascending=True)

    if top_k is not None:
        method_df = method_df.tail(top_k)

    plt.figure(figsize=(10, max(6, 0.45 * len(method_df))))
    plt.barh(method_df["method"], method_df["count"])
    plt.xlabel("Number of runs using method")
    plt.ylabel("Method")
    plt.title("Method usage across all runs" + (" (canonicalized)" if canonical else ""))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


# =========================================================
# Plot 2: number of methods vs accuracy
# =========================================================
def plot_num_methods_vs_accuracy(df, save_path=None, canonical=True):
    count_col = _get_n_method_col(canonical)

    plt.figure(figsize=(8, 6))
    plt.scatter(df[count_col], df["test_accuracy"], alpha=0.8)

    if len(df) >= 2:
        x = df[count_col].values
        y = df["test_accuracy"].values
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = m * x_line + b
        plt.plot(x_line, y_line)

    plt.xlabel("Number of methods used")
    plt.ylabel("Test accuracy")
    plt.title("Accuracy vs number of methods" + (" (canonicalized)" if canonical else ""))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    corr = df[count_col].corr(df["test_accuracy"])
    print(f"Pearson correlation between {count_col} and accuracy: {corr:.4f}")


# =========================================================
# Plot 3: top vs bottom method comparison
# =========================================================
def plot_top_bottom_method_comparison(df, save_path=None, k=10, canonical=True):
    method_col = _get_method_col(canonical)

    df_sorted = df.sort_values("test_accuracy", ascending=False).reset_index(drop=True)
    top_df = df_sorted.head(k)
    bottom_df = df_sorted.tail(k)

    top_counter = Counter()
    bottom_counter = Counter()

    for methods in top_df[method_col]:
        top_counter.update(methods)
    for methods in bottom_df[method_col]:
        bottom_counter.update(methods)

    all_methods = sorted(set(top_counter) | set(bottom_counter))
    comp_df = pd.DataFrame({
        "method": all_methods,
        "top_k_count": [top_counter[m] for m in all_methods],
        "bottom_k_count": [bottom_counter[m] for m in all_methods],
    })
    comp_df["diff"] = comp_df["top_k_count"] - comp_df["bottom_k_count"]
    comp_df = comp_df.sort_values("diff")

    plt.figure(figsize=(11, max(6, 0.45 * len(comp_df))))
    plt.barh(comp_df["method"], comp_df["bottom_k_count"], label=f"Bottom {k}", alpha=0.75)
    plt.barh(comp_df["method"], comp_df["top_k_count"], label=f"Top {k}", alpha=0.75)
    plt.xlabel("Count")
    plt.ylabel("Method")
    plt.title(f"Method usage in top {k} vs bottom {k} runs" + (" (canonicalized)" if canonical else ""))
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    print("\nMethods more common in top runs:")
    print(comp_df.sort_values("diff", ascending=False).head(10)[
        ["method", "top_k_count", "bottom_k_count", "diff"]
    ])

    print("\nMethods more common in bottom runs:")
    print(comp_df.sort_values("diff", ascending=True).head(10)[
        ["method", "top_k_count", "bottom_k_count", "diff"]
    ])


# =========================================================
# Plot 4: top and bottom runs by accuracy
# =========================================================
def plot_top_bottom_runs(df, save_path=None, k=10):
    df_sorted = df.sort_values("test_accuracy", ascending=False).reset_index(drop=True)
    top_df = df_sorted.head(k).copy()
    bottom_df = df_sorted.tail(k).copy()

    fig, axes = plt.subplots(2, 1, figsize=(11, 10))

    axes[0].barh(top_df["run_id"][::-1], top_df["test_accuracy"][::-1])
    axes[0].set_title(f"Top {k} runs by accuracy")
    axes[0].set_xlabel("Test accuracy")

    axes[1].barh(bottom_df["run_id"][::-1], bottom_df["test_accuracy"][::-1])
    axes[1].set_title(f"Bottom {k} runs by accuracy")
    axes[1].set_xlabel("Test accuracy")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


# =========================================================
# Plot 5: method heatmap for top/bottom runs
# =========================================================
def plot_method_heatmap_for_extremes(df, save_path=None, k=10, canonical=True, min_count=2):
    method_col = _get_method_col(canonical)

    df_sorted = df.sort_values("test_accuracy", ascending=False).reset_index(drop=True)
    extreme_df = pd.concat([df_sorted.head(k), df_sorted.tail(k)], axis=0).copy()

    method_counter = Counter()
    for methods in extreme_df[method_col]:
        method_counter.update(methods)

    methods_kept = [m for m, c in method_counter.items() if c >= min_count]
    methods_kept = sorted(methods_kept)

    matrix = []
    labels = []
    for _, row in extreme_df.iterrows():
        row_methods = set(row[method_col])
        matrix.append([1 if m in row_methods else 0 for m in methods_kept])
        labels.append(f"{row['run_id']} ({row['test_accuracy']:.3f})")

    matrix = np.array(matrix)

    plt.figure(figsize=(max(10, 0.35 * len(methods_kept)), max(6, 0.4 * len(labels))))
    plt.imshow(matrix, aspect="auto")
    plt.yticks(range(len(labels)), labels)
    plt.xticks(range(len(methods_kept)), methods_kept, rotation=90)
    plt.xlabel("Method")
    plt.ylabel("Run")
    plt.title(f"Method presence for top {k} and bottom {k} runs" + (" (canonicalized)" if canonical else ""))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


# =========================================================
# Plot 6: backbone usage
# =========================================================
def plot_backbone_usage(df, save_path=None):
    counts = df["backbone_bucket"].value_counts()

    plt.figure(figsize=(7, 5))
    plt.bar(counts.index, counts.values)
    plt.ylabel("Number of runs")
    plt.xlabel("Backbone usage")
    plt.title("DINOv2 vs CLIP usage across runs")
    plt.xticks(rotation=15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    print("\nBackbone usage counts:")
    print(counts)


# =========================================================
# Plot 7: LLM cost vs accuracy
# =========================================================
def plot_cost_vs_accuracy(df, save_path=None):
    if df["llm_cost"].isna().all():
        print("Skipping cost plot because llm_cost is unavailable.")
        return

    plt.figure(figsize=(8, 6))
    plt.scatter(df["llm_cost"], df["test_accuracy"], alpha=0.8)

    if len(df) >= 2:
        x = df["llm_cost"].values
        y = df["test_accuracy"].values
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = m * x_line + b
        plt.plot(x_line, y_line)

    plt.xlabel("LLM cost per run ($)")
    plt.ylabel("Test accuracy")
    plt.title("Accuracy vs LLM cost")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    corr = df["llm_cost"].corr(df["test_accuracy"])
    print(f"Pearson correlation between llm_cost and accuracy: {corr:.4f}")


def print_cost_summary(df):
    if df["llm_cost"].isna().all():
        print("No LLM pricing provided, so cost summary was skipped.")
        return

    print("\n===== LLM Cost Summary =====")
    print(f"Number of runs: {len(df)}")
    print(f"Total cost: ${df['llm_cost'].sum():.6f}")
    print(f"Mean cost/run: ${df['llm_cost'].mean():.6f}")
    print(f"Median cost/run: ${df['llm_cost'].median():.6f}")
    print(f"Min cost/run: ${df['llm_cost'].min():.6f}")
    print(f"Max cost/run: ${df['llm_cost'].max():.6f}")


# =========================================================
# Main
# =========================================================
def main():
    # -----------------------------------------------------
    # Edit these
    # -----------------------------------------------------
    input_path = "/home/eyl45/Sun/aira-dojo/src/dojo/analysis/runs/dtd/aSSL_backbone_unsupervised/k3/seed0/aira/2/analysis_dict.json"
    output_dir = Path("viz_outputs/dtd/unsupervised/k3/")
    os.makedirs(output_dir, exist_ok=True)

    # Set these if you want cost calculations.
    # Example placeholder values:
    prompt_price_per_1m = 1.10
    completion_price_per_1m = 4.40

    # Fuzzy threshold:
    fuzzy_threshold = 70

    # -----------------------------------------------------
    # Load and preprocess
    # -----------------------------------------------------
    results = load_results(input_path)

    df = results_to_dataframe(
        results,
        prompt_price_per_1m=prompt_price_per_1m,
        completion_price_per_1m=completion_price_per_1m,
    )

    df, raw_to_final, clusters = add_canonical_methods(df, threshold=fuzzy_threshold)

    print(f"Loaded {len(df)} runs.")
    print(df[[
        "run_id", "test_accuracy", "n_methods", "n_methods_canonical", "backbone_bucket"
    ]].head())

    # Save flattened CSV
    df.to_csv(output_dir / "flattened_results.csv", index=False)

    # Print fuzzy groupings so you can inspect them
    print_method_clusters(clusters)

    print("\n===== Raw -> Canonical Method Mapping =====")
    for raw, canon in sorted(raw_to_final.items()):
        print(f"{raw}  -->  {canon}")

    # Cost summary
    print_cost_summary(df)
    print("\nTotal experiment cost:")
    print(
        total_experiment_cost(
            results,
            prompt_price_per_1m=prompt_price_per_1m,
            completion_price_per_1m=completion_price_per_1m,
        )
    )

    # -----------------------------------------------------
    # Plots
    # -----------------------------------------------------
    plot_method_frequency(
        df,
        save_path=output_dir / "method_frequency_canonical.png",
        canonical=True,
    )

    plot_num_methods_vs_accuracy(
        df,
        save_path=output_dir / "num_methods_vs_accuracy_canonical.png",
        canonical=True,
    )

    plot_top_bottom_method_comparison(
        df,
        save_path=output_dir / "top_bottom_method_comparison_canonical.png",
        k=10,
        canonical=True,
    )

    plot_top_bottom_runs(
        df,
        save_path=output_dir / "top_bottom_runs.png",
        k=10,
    )

    plot_method_heatmap_for_extremes(
        df,
        save_path=output_dir / "top_bottom_method_heatmap_canonical.png",
        k=10,
        canonical=True,
        min_count=2,
    )

    plot_backbone_usage(
        df,
        save_path=output_dir / "backbone_usage.png",
    )

    plot_cost_vs_accuracy(
        df,
        save_path=output_dir / "cost_vs_accuracy.png",
    )


if __name__ == "__main__":
    main()