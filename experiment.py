"""Two pre-specified experiments about decision trees and incorrect labels.

1. Can limiting tree depth improve robustness to training-label noise?
2. Can tree depth be selected reliably when validation labels are also noisy?

Run this file with Python 3.12+ after installing numpy, pandas, matplotlib,
and scikit-learn. It installs nothing and requires no other project files.
The original experiment writes two CSVs and eight PNGs under outputs/; the validation
experiment writes three CSVs and five PNGs under outputs/validation_noise/.
Figures 06-13 are additional descriptive views of the same saved results, added
after the experiments. They introduce no new model fits or significance tests.
Experiment 1 retains 20 repetitions (320 fits). Experiment 2 was expanded after
the initial run to 30 repetitions (480 fits and 480 selections), keeping all
original seeds and adding seeds 20-29. This is a documented design revision.
All paths are relative to this file. Rerunning replaces these eighteen files.
Package versions are printed for your records;
use the same versions for the closest reproducibility across computers.
Verified environment: Python 3.12.14, NumPy 2.5.4, pandas 3.0.6,
Matplotlib 3.11.2, scikit-learn 1.9.1.

This is an AI-assisted experiment, not a claim of an original theoretical result.
Read the comments and inspect the results before drawing your own conclusions.
"""

from inspect import signature
from itertools import combinations, product
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib

# Select the non-interactive backend BEFORE importing pyplot: no windows to close.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import sklearn
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# --------------------- Design fixed before inspecting results -----------------
N_SAMPLES = 1_000
FEATURE_NOISE = 0.20
TEST_SIZE = 0.30
REPETITION_SEEDS = tuple(range(20))
NOISE_RATES = (0.00, 0.10, 0.20, 0.30)
MAX_DEPTHS = (2, 4, 8, None)
ILLUSTRATION_SEED = 0  # Selected in advance, never chosen for a favorable picture.
ILLUSTRATION_CONDITIONS = ((0.00, 2), (0.00, None), (0.30, 2), (0.30, None))
EXPECTED_FITS = 320
EXPECTED_SUMMARY_ROWS = 16

# A tree predicts by repeatedly asking questions such as "is feature 1 <= 0.5?".
# max_depth limits the number of splitting steps along a path through the tree.
# A very shallow tree may underfit: it cannot represent the curved moon shapes well.
# A deeper tree can fit finer details, including incorrect labels. Greater training
# accuracy therefore need not mean better prediction on new samples.
# Only max_depth varies. All remaining classifier settings are fixed explicitly.
TREE_SETTINGS = {
    "criterion": "gini",
    "splitter": "best",
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": None,
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0.0,
    "class_weight": None,
    "ccp_alpha": 0.0,
    "monotonic_cst": None,
}

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
RAW_PATH = OUTPUT_DIR / "raw_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary_results.csv"
FIGURE_PATHS = (
    FIGURE_DIR / "01_test_accuracy_vs_noise.png",
    FIGURE_DIR / "02_training_vs_test_accuracy.png",
    FIGURE_DIR / "03_decision_boundaries.png",
)
FIGURE_DPI = 300
DEPTH_LABELS = ("2", "4", "8", "unlimited")
DEPTH_COLORS = {"2": "#0072B2", "4": "#E69F00", "8": "#009E73", "unlimited": "#CC79A7"}
CLASS_COLORS = ("#0072B2", "#D55E00")
BOUNDARY_GRID_SIZE = 350
BOUNDARY_MARGIN = 0.25
ACCURACY_Y_LIMITS = (0.0, 1.05)  # Full accuracy range; small space above 100%.
ACCURACY_Y_TICKS = np.linspace(0.0, 1.0, 6)
NUMERIC_TOLERANCE = 1e-12
ACCURACY_COLUMNS = ("train_accuracy_noisy", "train_accuracy_clean", "test_accuracy")
PAIRED_COLUMN = "test_accuracy_difference_vs_unlimited"
SUMMARY_METRICS = (*ACCURACY_COLUMNS, "generalization_gap_noisy", PAIRED_COLUMN)
RAW_COLUMNS = (
    "seed", "requested_noise_rate", "actual_noise_rate", "n_flipped", "max_depth",
    "n_train", "n_test", *ACCURACY_COLUMNS, "generalization_gap_noisy",
    "fitted_tree_depth", "n_leaves", PAIRED_COLUMN,
)

# Experiment 2 shares the data mechanism and tree settings, but uses its OWN split
# and random-stream namespace. Never pool the experiments into one comparison.
SELECTION_N_SAMPLES = 1_000
SELECTION_FEATURE_NOISE = 0.20
# Revision: expand experiment 2 from 20 to 30 simulations after the initial run.
# Retain seeds 0-19 and add all seeds 20-29; do not select seeds by their results.
SELECTION_SEEDS = tuple(range(30))
SELECTION_STREAM_NAMESPACE = 1
SELECTION_HOLDOUT_FRACTION = 0.40  # First split: 600 training / 400 held out.
SELECTION_TEST_SHARE_OF_HOLDOUT = 0.50  # Second split: 200 validation / 200 test.
SELECTION_TRAIN_NOISE_RATES = (0.00, 0.10, 0.20, 0.30)
SELECTION_VALIDATION_NOISE_RATES = (0.00, 0.10, 0.20, 0.30)
SELECTION_DEPTHS = MAX_DEPTHS
SELECTION_TIE_ORDER = DEPTH_LABELS  # First maximum wins: 2, 4, 8, unlimited.
SELECTION_EXPECTED_FITS = len(SELECTION_SEEDS) * len(SELECTION_TRAIN_NOISE_RATES) * len(SELECTION_DEPTHS)
SELECTION_EXPECTED_OUTCOMES = len(SELECTION_SEEDS) * len(SELECTION_TRAIN_NOISE_RATES) * len(SELECTION_VALIDATION_NOISE_RATES)
SELECTION_EXPECTED_GROUPS = 16
SELECTION_EXPECTED_FREQUENCY_ROWS = 64
SELECTION_OUTPUT_DIR = OUTPUT_DIR / "validation_noise"
SELECTION_FIGURE_DIR = SELECTION_OUTPUT_DIR / "figures"
SELECTION_RAW_PATH = SELECTION_OUTPUT_DIR / "selection_raw.csv"
SELECTION_SUMMARY_PATH = SELECTION_OUTPUT_DIR / "selection_summary.csv"
SELECTION_FREQUENCIES_PATH = SELECTION_OUTPUT_DIR / "selection_frequencies.csv"
SELECTION_FIGURE_PATHS = (
    SELECTION_FIGURE_DIR / "04_validation_noise_test_change.png",
    SELECTION_FIGURE_DIR / "05_depth_selection_frequencies.png",
)
SELECTION_GROUP_KEYS = ("requested_train_noise_rate", "requested_validation_noise_rate")
SELECTION_CHANGE = "test_accuracy_change_vs_clean_validation"
SELECTION_CHANGED = "selection_changed_vs_clean_validation"
SELECTION_ACCURACIES = (
    "selected_validation_accuracy_observed", "selected_validation_accuracy_clean",
    "selected_test_accuracy", "reference_test_accuracy_clean_validation",
)
SELECTION_RAW_COLUMNS = (
    "seed", "requested_train_noise_rate", "actual_train_noise_rate",
    "requested_validation_noise_rate", "actual_validation_noise_rate",
    "n_train_flipped", "n_validation_flipped", "n_train", "n_validation", "n_test",
    "selected_depth", "selected_validation_accuracy_observed",
    "selected_validation_accuracy_clean", "selected_test_accuracy",
    "reference_depth_clean_validation", "reference_test_accuracy_clean_validation",
    SELECTION_CHANGED, SELECTION_CHANGE,
)

# Additional plots use the existing results only; the experimental design above
# stays fixed. Figure 10 shows the largest studied validation flip rate, 30%.
ADDITIONAL_TRAIN_FIGURE_PATHS = (
    FIGURE_DIR / "06_paired_advantage_vs_unlimited.png",
    FIGURE_DIR / "07_test_change_from_unflipped_training.png",
    FIGURE_DIR / "08_fitted_tree_complexity.png",
    FIGURE_DIR / "12_limited_depth_win_tie_loss.png",
    FIGURE_DIR / "13_original_training_vs_test_accuracy.png",
)
ADDITIONAL_SELECTION_FIGURE_PATHS = (
    SELECTION_FIGURE_DIR / "09_test_accuracy_with_clean_validation_reference.png",
    SELECTION_FIGURE_DIR / "10_depth_selection_transitions.png",
    SELECTION_FIGURE_DIR / "11_test_performance_outcome_frequencies.png",
)
TRANSITION_VALIDATION_RATE = 0.30
PERFORMANCE_OUTCOME_COLORS = ("#D55E00", "#B8BDC5", "#009E73")

# Training data teaches each candidate. Validation data selects a candidate depth.
# Test data evaluates the selection procedure without participating in selection.
# Noisy validation labels can make selection less reliable, but a changed choice
# need not perform worse: it may do better on this test set by chance.
# Original validation labels are a simulation reference, not an oracle or information
# normally available when real validation labels are corrupted. The fixed preference
# for shallower trees in exact ties can affect depth-selection frequencies.
# Both experiments concern one synthetic mechanism and symmetric random label flips.

# Feature noise perturbs the input coordinates. Label noise changes class labels.
# The original make_moons labels are the reference labels. "Clean" below means
# not deliberately flipped; feature noise still makes the task ambiguous.
# Experiment 1 uses 20 simulations; experiment 2 now uses 30. Both sample ONE
# synthetic mechanism, rather than distinct real-world datasets.
# Repetitions show how results vary when samples, splits, and noise assignments vary.
# Conditions share samples and flip sets within a repetition so differences are paired.
# Experiment 1 compares fixed depths without selecting a model. Neither experiment
# discards repetitions or changes its design after inspecting test results.
# Selecting a model using test results would not give that model an unbiased test;
# Experiment 2 therefore selects using validation labels only.
# Limits: one synthetic mechanism; random symmetric training-label flips; fixed
# sample size and feature noise; only four depths. The best average here need not
# generalize to other datasets. No post-hoc significance tests are performed.


def require(condition, message):
    """Raise a helpful error even when Python is run with assertions disabled."""
    if not condition:
        raise ValueError(f"Validation failed: {message}")


def depth_label(max_depth):
    return "unlimited" if max_depth is None else str(max_depth)


def make_random_seeds(seed):
    """Derive separate, reproducible streams without using global random state."""
    children = np.random.SeedSequence(seed).spawn(4)
    names = ("data", "split", "flips", "model")
    return {
        name: int(child.generate_state(1, dtype=np.uint32)[0])
        for name, child in zip(names, children)
    }


def generate_and_split_data(random_seeds):
    """Generate a new dataset and a stratified 700/300 split using sample indices."""
    X, y = make_moons(
        n_samples=N_SAMPLES, noise=FEATURE_NOISE, shuffle=True,
        random_state=random_seeds["data"],
    )
    train_indices, test_indices = train_test_split(
        np.arange(N_SAMPLES), test_size=TEST_SIZE, stratify=y, shuffle=True,
        random_state=random_seeds["split"],
    )
    require(np.intersect1d(train_indices, test_indices).size == 0,
            "training and test indices overlap.")
    require(np.array_equal(np.sort(np.concatenate([train_indices, test_indices])),
                           np.arange(N_SAMPLES)), "the split loses or duplicates samples.")
    require(len(train_indices) == 700 and len(test_indices) == 300,
            "the pre-specified split must contain 700 training and 300 test samples.")
    require(X.shape == (N_SAMPLES, 2) and np.isfinite(X).all(),
            "features must be finite with shape (1000, 2).")
    for indices, name in ((train_indices, "training"), (test_indices, "test")):
        require(np.array_equal(np.bincount(y[indices], minlength=2),
                               [len(indices) // 2, len(indices) // 2]),
                f"{name} class proportions do not match the stratified design.")
    # Trees split individual features at thresholds, so scaling is unnecessary.
    data = {
        "X_train": X[train_indices].copy(),
        "X_test": X[test_indices].copy(),
        "y_train_clean": y[train_indices].copy(),
        "y_test": y[test_indices].copy(),
        "train_indices": train_indices.copy(),
        "test_indices": test_indices.copy(),
    }
    # Read-only arrays prevent accidental in-place changes; snapshots below also
    # detect replacement with different arrays between conditions.
    for array in data.values():
        array.setflags(write=False)
    return data


def flip_binary_labels(y_clean, noise_rate, permutation, subset_name):
    """Shared implementation for training/validation labels; never used on test labels."""
    n_labels = len(y_clean)
    require(0.0 <= noise_rate <= 1.0, "requested flip rate is outside [0, 1].")
    require(np.isin(y_clean, [0, 1]).all(), "label flipping requires binary 0/1 labels.")
    require(np.array_equal(np.sort(permutation), np.arange(n_labels)),
            f"flip order is not a permutation of all {subset_name} indices.")
    original = y_clean.copy()
    k = int(round(noise_rate * n_labels))
    flipped_indices = permutation[:k].copy()
    y_noisy = y_clean.copy()  # Preserve the original reference labels.
    y_noisy[flipped_indices] = 1 - y_noisy[flipped_indices]
    require(np.array_equal(y_clean, original), f"original {subset_name} labels were modified.")
    require(np.count_nonzero(y_noisy != y_clean) == k, "incorrect number of flipped labels.")
    require(np.array_equal(np.flatnonzero(y_noisy != y_clean), np.sort(flipped_indices)),
            "labels changed outside the intended flipped indices.")
    y_noisy.setflags(write=False)
    return y_noisy, flipped_indices, k / n_labels


def flip_training_labels(y_clean, noise_rate, permutation):
    """Keep the original experiment's interface and numerical behavior unchanged."""
    return flip_binary_labels(y_clean, noise_rate, permutation, "training")


def validate_condition_inputs(data, reference_data, y_noisy, reference_y, context):
    """Check the actual inputs used at every fit, before and after that fit."""
    for name, expected in reference_data.items():
        require(np.array_equal(data[name], expected),
                f"{context}: {name} changed between conditions (including reference labels).")
    require(np.array_equal(y_noisy, reference_y),
            f"{context}: models within one noise condition received different labels.")


def fit_and_evaluate_model(data, y_noisy, max_depth, model_seed,
                           reference_data, reference_y, context):
    """Fit once and score three ways, using only training data in model.fit()."""
    validate_condition_inputs(data, reference_data, y_noisy, reference_y, context)
    model = DecisionTreeClassifier(
        max_depth=max_depth, random_state=model_seed, **TREE_SETTINGS,
    )
    model.fit(data["X_train"], y_noisy)  # One fit per original-experiment condition.
    train_predictions = model.predict(data["X_train"])
    test_predictions = model.predict(data["X_test"])
    metrics = {
        "train_accuracy_noisy": accuracy_score(y_noisy, train_predictions),
        "train_accuracy_clean": accuracy_score(data["y_train_clean"], train_predictions),
        "test_accuracy": accuracy_score(data["y_test"], test_predictions),
        "fitted_tree_depth": model.get_depth(),
        "n_leaves": model.get_n_leaves(),
    }
    metrics["generalization_gap_noisy"] = (
        metrics["train_accuracy_noisy"] - metrics["test_accuracy"]
    )
    # These two accuracies use DIFFERENT label references. Their gap alone does
    # not prove memorization of incorrect labels: also inspect clean train/test
    # accuracy. Test labels stay untouched to measure prediction against reference
    # labels, rather than mixing training corruption with test-label corruption.
    validate_condition_inputs(data, reference_data, y_noisy, reference_y, context)
    require(max_depth is None or model.get_depth() <= max_depth,
            f"{context}: fitted tree exceeds its depth limit.")
    return model, metrics


def run_repetitions():
    """Run all 320 fits, retaining four models for the pre-specified illustration."""
    rows = []
    illustration = {}
    fit_count = 0
    for repetition_number, seed in enumerate(REPETITION_SEEDS, start=1):
        random_seeds = make_random_seeds(seed)
        data = generate_and_split_data(random_seeds)
        reference_data = {name: array.copy() for name, array in data.items()}
        # This reruns data generation and splitting, but does NOT fit more models.
        replay = generate_and_split_data(make_random_seeds(seed))
        for name in data:
            require(np.array_equal(data[name], replay[name]),
                    f"seed {seed}: regenerating {name} did not reproduce identical arrays.")
        n_train, n_test = len(data["y_train_clean"]), len(data["y_test"])
        permutation = np.random.default_rng(random_seeds["flips"]).permutation(n_train)
        replay_permutation = np.random.default_rng(
            make_random_seeds(seed)["flips"]
        ).permutation(n_train)
        require(np.array_equal(permutation, replay_permutation),
                f"seed {seed}: flip permutation is not reproducible.")
        previous_flips = set()
        for noise_rate in NOISE_RATES:
            y_noisy, flipped_indices, actual_rate = flip_training_labels(
                data["y_train_clean"], noise_rate, permutation,
            )
            replay_y, replay_indices, replay_rate = flip_training_labels(
                replay["y_train_clean"], noise_rate, replay_permutation,
            )
            require(np.array_equal(y_noisy, replay_y)
                    and np.array_equal(flipped_indices, replay_indices)
                    and actual_rate == replay_rate,
                    f"seed {seed}, noise {noise_rate}: modified labels are not reproducible.")
            current_flips = set(flipped_indices.tolist())
            require(previous_flips.issubset(current_flips),
                    f"seed {seed}: flipped index sets are not nested.")
            previous_flips = current_flips
            reference_y = y_noisy.copy()
            for max_depth in MAX_DEPTHS:
                label = depth_label(max_depth)
                context = f"seed {seed}, noise {noise_rate}, depth {label}"
                model, metrics = fit_and_evaluate_model(
                    data, y_noisy, max_depth, random_seeds["model"],
                    reference_data, reference_y, context,
                )
                fit_count += 1
                rows.append({
                    "seed": seed, "requested_noise_rate": noise_rate,
                    "actual_noise_rate": actual_rate, "n_flipped": len(flipped_indices),
                    "max_depth": label, "n_train": n_train, "n_test": n_test, **metrics,
                })
                if seed == ILLUSTRATION_SEED and (noise_rate, max_depth) in ILLUSTRATION_CONDITIONS:
                    illustration[(noise_rate, label)] = {
                        "model": model, "data": data, "y_noisy": y_noisy,
                        "flipped_indices": flipped_indices.copy(),
                        "test_accuracy": metrics["test_accuracy"],
                    }
        print(f"Repetition {repetition_number:02d}/20 (seed={seed:02d}): "
              f"16 fits; input, flip, and reproducibility checks passed.", flush=True)
    require(fit_count == EXPECTED_FITS, f"expected 320 fits, got {fit_count}.")
    require(len(illustration) == 4, "expected four retained models for Figure 3.")
    raw = pd.DataFrame(rows)
    # Pair by BOTH seed and noise rate before taking any means or standard deviations.
    baseline = raw.loc[raw["max_depth"] == "unlimited",
                       ["seed", "requested_noise_rate", "test_accuracy"]].rename(
        columns={"test_accuracy": "unlimited_test_accuracy"}
    )
    raw = raw.merge(baseline, on=["seed", "requested_noise_rate"],
                    how="left", validate="many_to_one", sort=False)
    raw[PAIRED_COLUMN] = raw["test_accuracy"] - raw["unlimited_test_accuracy"]
    return raw.loc[:, list(RAW_COLUMNS)], illustration


def validate_results(raw, summary=None, announce=True):
    """Check completeness, numeric validity, pairing, and (if supplied) summaries."""
    require(tuple(raw.columns) == RAW_COLUMNS, "raw CSV columns do not match the design.")
    keys = ["seed", "requested_noise_rate", "max_depth"]
    expected_keys = set(product(REPETITION_SEEDS, NOISE_RATES, DEPTH_LABELS))
    actual_keys = set(raw[keys].itertuples(index=False, name=None))
    require(len(raw) == EXPECTED_FITS and not raw.duplicated(keys).any()
            and actual_keys == expected_keys,
            "raw results must contain exactly 320 unique pre-specified combinations.")
    numeric_columns = [name for name in RAW_COLUMNS if name != "max_depth"]
    require(np.isfinite(raw[numeric_columns].to_numpy(dtype=float)).all(),
            "raw numeric results contain missing or infinite values.")
    for name in ACCURACY_COLUMNS:
        require(raw[name].between(0.0, 1.0).all(), f"{name} is outside [0, 1].")
    require((raw["n_train"] == 700).all() and (raw["n_test"] == 300).all(),
            "recorded training or test sample counts are incorrect.")
    expected_flips = [int(round(rate * n)) for rate, n in
                      zip(raw["requested_noise_rate"], raw["n_train"])]
    require(np.array_equal(raw["n_flipped"].to_numpy(), expected_flips),
            "recorded flip counts do not equal round(rate * n_train).")
    require(np.allclose(raw["actual_noise_rate"], raw["n_flipped"] / raw["n_train"],
                        rtol=0, atol=NUMERIC_TOLERANCE), "actual flip rates are incorrect.")
    require(np.allclose(raw["generalization_gap_noisy"],
                        raw["train_accuracy_noisy"] - raw["test_accuracy"],
                        rtol=0, atol=NUMERIC_TOLERANCE), "recorded generalization gaps are incorrect.")
    # An independent, keyed lookup checks the merge used to calculate paired values.
    baselines = raw.loc[raw["max_depth"] == "unlimited"].set_index(
        ["seed", "requested_noise_rate"]
    )["test_accuracy"]
    for row in raw.itertuples(index=False):
        expected_difference = row.test_accuracy - baselines.loc[
            (row.seed, row.requested_noise_rate)
        ]
        require(np.isclose(getattr(row, PAIRED_COLUMN), expected_difference,
                           rtol=0, atol=NUMERIC_TOLERANCE),
                f"seed {row.seed}, noise {row.requested_noise_rate}: comparison uses wrong baseline.")
        require(row.fitted_tree_depth >= 0 and row.fitted_tree_depth == int(row.fitted_tree_depth)
                and 1 <= row.n_leaves <= row.n_train and row.n_leaves == int(row.n_leaves),
                "invalid fitted tree depth or leaf count.")
        if row.max_depth != "unlimited":
            require(row.fitted_tree_depth <= int(row.max_depth), "recorded depth exceeds its limit.")
    require((raw.loc[raw["max_depth"] == "unlimited", PAIRED_COLUMN] == 0.0).all(),
            "unlimited models must have paired differences of exactly zero.")
    if announce:
        print("Validation passed: 320 unique fits, finite metrics, accuracy bounds, and correct pairing.")
    if summary is None:
        return

    summary_keys = ["requested_noise_rate", "max_depth"]
    require(len(summary) == EXPECTED_SUMMARY_ROWS
            and not summary.duplicated(summary_keys).any()
            and set(summary[summary_keys].itertuples(index=False, name=None))
            == set(product(NOISE_RATES, DEPTH_LABELS))
            and (summary["repetition_count"] == 20).all(),
            "summary must contain exactly 16 groups, each with 20 repetitions.")
    numeric_columns = [name for name in summary.columns if name != "max_depth"]
    require(np.isfinite(summary[numeric_columns].to_numpy(dtype=float)).all(),
            "summary numeric results contain missing or infinite values.")
    for name in ACCURACY_COLUMNS:
        require(summary[f"{name}_mean"].between(0.0, 1.0).all(),
                f"summary mean {name} is outside [0, 1].")
    for row in summary.to_dict("records"):
        group = raw.loc[(raw["requested_noise_rate"] == row["requested_noise_rate"])
                        & (raw["max_depth"] == row["max_depth"])]
        require(len(group) == 20 and set(group["seed"]) == set(REPETITION_SEEDS),
                "a summary group is missing or duplicates repetition seeds.")
        for metric in SUMMARY_METRICS:
            values = group[metric].to_numpy()
            for suffix, expected in (("mean", np.mean(values)), ("std", np.std(values, ddof=1))):
                require(np.isclose(row[f"{metric}_{suffix}"], expected,
                                   rtol=0, atol=NUMERIC_TOLERANCE),
                        f"incorrect summary {metric}_{suffix}; use sample SD with ddof=1.")
    if announce:
        print("Validation passed: 16 complete groups of 20; means and sample SDs verified (ddof=1).")


def summarize_results(raw):
    """Describe all repetitions; SD is simulation-to-simulation variation, not a CI."""
    rows = []
    for noise_rate, label in product(NOISE_RATES, DEPTH_LABELS):
        group = raw.loc[(raw["requested_noise_rate"] == noise_rate)
                        & (raw["max_depth"] == label)]
        row = {
            "requested_noise_rate": noise_rate, "max_depth": label,
            "repetition_count": group["seed"].nunique(),
        }
        for metric in SUMMARY_METRICS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def format_accuracy_axis(ax):
    """Keep the units and ranges consistent across accuracy plots."""
    ax.set_xticks(NOISE_RATES)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(*ACCURACY_Y_LIMITS)
    ax.set_yticks(ACCURACY_Y_TICKS)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.grid(True, alpha=0.20)
    ax.set_axisbelow(True)


def create_figure_1(summary):
    """Mean test accuracy with +/- one sample SD, for all four depths."""
    fig, ax = plt.subplots(figsize=(10, 6.7))
    for label, marker in zip(DEPTH_LABELS, ("o", "s", "^", "D")):
        group = summary.loc[summary["max_depth"] == label].sort_values("requested_noise_rate")
        ax.errorbar(
            group["requested_noise_rate"], group["test_accuracy_mean"],
            yerr=group["test_accuracy_std"], label=f"Depth {label}",
            color=DEPTH_COLORS[label], marker=marker, linewidth=2,
            capsize=4, elinewidth=1.2, markersize=6,
        )
    format_accuracy_axis(ax)
    ax.set_title("Test Accuracy vs Training Label Noise", fontsize=16, pad=14)
    ax.set_xlabel("Training labels flipped (%)")
    ax.set_ylabel("Mean test accuracy (%)")
    ax.legend(loc="lower left", ncols=4, title="Maximum tree depth", fontsize=10)
    fig.text(0.5, 0.055, "20 repeated simulations; test labels were never flipped.\n"
             "Error bars: mean ± one sample standard deviation (not a confidence interval).",
             ha="center", va="center", fontsize=10)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.22)
    fig.savefig(FIGURE_PATHS[0], dpi=FIGURE_DPI)
    plt.close(fig)


def create_figure_2(summary):
    """Use depth colors consistently; line styles distinguish reference labels."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9), sharex=True, sharey=True)
    styles = (
        ("train_accuracy_noisy", "Training: noisy labels", "--", "s"),
        ("train_accuracy_clean", "Training: original labels", ":", "^"),
        ("test_accuracy", "Test: untouched labels", "-", "o"),
    )
    for ax, label in zip(axes.flat, DEPTH_LABELS):
        group = summary.loc[summary["max_depth"] == label].sort_values("requested_noise_rate")
        for metric, description, linestyle, marker in styles:
            ax.plot(group["requested_noise_rate"], group[f"{metric}_mean"],
                    label=description, color=DEPTH_COLORS[label], linestyle=linestyle,
                    marker=marker, linewidth=2, markersize=5)
        format_accuracy_axis(ax)
        ax.set_title(f"Maximum depth: {label}", color=DEPTH_COLORS[label], fontweight="bold")
        ax.tick_params(labelbottom=True, labelleft=True)
    # A neutral legend explains line styles; panel titles identify depth colors.
    handles = [Line2D([0], [0], color="#333333", linestyle=style, marker=marker,
                      linewidth=2, label=description)
               for _, description, style, marker in styles]
    fig.suptitle("Training and Test Accuracy", fontsize=17, y=0.97)
    fig.supxlabel("Training labels flipped (%)", y=0.175)
    fig.supylabel("Mean accuracy (%)", x=0.018, y=0.58)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.105),
               ncols=3, frameon=False, fontsize=10)
    fig.text(0.5, 0.052, "Means over 20 simulations. Noisy training accuracy uses the labels supplied to fit the model.\n"
             "Original training and untouched test labels are reference labels. The noisy train–test gap alone\n"
             "does not establish memorization; inspect all three accuracies together.",
             ha="center", va="center", fontsize=10)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.25, hspace=0.25, wspace=0.12)
    fig.savefig(FIGURE_PATHS[1], dpi=FIGURE_DPI)
    plt.close(fig)


def create_figure_3(illustration):
    """Predict on a plotting grid with the four retained models; do not refit."""
    first = illustration[(0.00, "2")]
    X_train = first["data"]["X_train"]
    # Common limits use training features only; the same grid serves all panels.
    x_limits = (X_train[:, 0].min() - BOUNDARY_MARGIN, X_train[:, 0].max() + BOUNDARY_MARGIN)
    y_limits = (X_train[:, 1].min() - BOUNDARY_MARGIN, X_train[:, 1].max() + BOUNDARY_MARGIN)
    xx, yy = np.meshgrid(np.linspace(*x_limits, BOUNDARY_GRID_SIZE),
                         np.linspace(*y_limits, BOUNDARY_GRID_SIZE))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    class_cmap = ListedColormap(CLASS_COLORS)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), sharex=True, sharey=True)
    for ax, (noise_rate, max_depth) in zip(axes.flat, ILLUSTRATION_CONDITIONS):
        label = depth_label(max_depth)
        example = illustration[(noise_rate, label)]
        for name, array in first["data"].items():
            require(np.array_equal(example["data"][name], array),
                    f"Figure 3: {name} differs between illustrated conditions.")
        regions = example["model"].predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, regions, levels=[-0.5, 0.5, 1.5], cmap=class_cmap, alpha=0.16)
        ax.scatter(X_train[:, 0], X_train[:, 1], c=example["y_noisy"],
                   cmap=class_cmap, vmin=0, vmax=1, s=16, alpha=0.80,
                   linewidths=0.25, edgecolors="white", zorder=2)
        flipped = example["flipped_indices"]
        if len(flipped):
            ax.scatter(X_train[flipped, 0], X_train[flipped, 1],
                       marker="x", color="black", s=30, linewidths=0.85, zorder=3)
        ax.set_title(f"Depth {label} | {noise_rate:.0%} label flips",
                     color=DEPTH_COLORS[label], fontweight="bold")
        ax.text(0.025, 0.97, f"Test accuracy: {example['test_accuracy']:.1%}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.93})
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.tick_params(labelbottom=True, labelleft=True)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
                      markeredgecolor="white", markersize=7, label=f"Training label {i}")
               for i, color in enumerate(CLASS_COLORS)]
    handles.append(Line2D([0], [0], marker="x", color="black", linestyle="none",
                          markersize=7, label="Flipped training label"))
    fig.suptitle("Illustrative Decision Boundaries", fontsize=17, y=0.97)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.10),
               ncols=3, frameon=False)
    fig.text(0.5, 0.045, "Pre-selected repetition seed = 0; identical samples and axis limits in every panel.\n"
             "Regions show predictions; point colors show labels used for training; black × marks label flips.\n"
             "This one illustration does not replace the aggregate results. Test labels were never flipped.",
             ha="center", va="center", fontsize=10)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.20, hspace=0.27, wspace=0.20)
    fig.savefig(FIGURE_PATHS[2], dpi=FIGURE_DPI)
    plt.close(fig)


def validate_output_files():
    """Fail clearly if writing any required deliverable was unsuccessful."""
    for path in (RAW_PATH, SUMMARY_PATH, *FIGURE_PATHS):
        require(path.is_file() and path.stat().st_size > 0,
                f"expected output is missing or empty: {path}")
    saved_raw = pd.read_csv(RAW_PATH, dtype={"max_depth": str})
    saved_summary = pd.read_csv(SUMMARY_PATH, dtype={"max_depth": str})
    require(len(saved_raw) == EXPECTED_FITS and len(saved_summary) == EXPECTED_SUMMARY_ROWS,
            "saved CSV row counts do not match the validated results.")
    validate_results(saved_raw, saved_summary, announce=False)
    print("Validation passed: both CSVs pass checks after reloading; all three PNGs exist and are nonempty.")


def make_selection_random_seeds(seed):
    """Five separate streams; the splitting stream has a child for each stage."""
    children = np.random.SeedSequence([seed, SELECTION_STREAM_NAMESPACE]).spawn(5)
    data, splitting, train_flips, validation_flips, model = children
    split_first, split_second = splitting.spawn(2)
    streams = {
        "data": data, "split_first": split_first, "split_second": split_second,
        "train_flips": train_flips, "validation_flips": validation_flips, "model": model,
    }
    seeds = {name: int(child.generate_state(1, dtype=np.uint32)[0])
             for name, child in streams.items()}
    require(len(set(seeds.values())) == len(seeds), "selection random seeds collide.")
    return seeds


def generate_selection_data(random_seeds):
    """Create and verify the two-stage stratified 600/200/200 split."""
    X, y = make_moons(n_samples=SELECTION_N_SAMPLES, noise=SELECTION_FEATURE_NOISE,
                      shuffle=True, random_state=random_seeds["data"])
    train_indices, holdout_indices = train_test_split(
        np.arange(SELECTION_N_SAMPLES), test_size=SELECTION_HOLDOUT_FRACTION,
        stratify=y, shuffle=True, random_state=random_seeds["split_first"],
    )
    validation_indices, test_indices = train_test_split(
        holdout_indices, test_size=SELECTION_TEST_SHARE_OF_HOLDOUT,
        stratify=y[holdout_indices], shuffle=True, random_state=random_seeds["split_second"],
    )
    partitions = {"train": train_indices, "validation": validation_indices, "test": test_indices}
    require(tuple(map(len, partitions.values())) == (600, 200, 200),
            "selection split sizes must be 600/200/200.")
    for left, right in combinations(partitions, 2):
        require(np.intersect1d(partitions[left], partitions[right]).size == 0,
                f"selection {left} and {right} indices overlap.")
    require(np.array_equal(np.sort(np.concatenate(list(partitions.values()))),
                           np.arange(SELECTION_N_SAMPLES)), "selection split loses or duplicates samples.")
    require(X.shape == (SELECTION_N_SAMPLES, 2) and np.isfinite(X).all(),
            "selection features have an invalid shape or nonfinite values.")
    data = {}
    for subset, indices in partitions.items():
        require(np.array_equal(np.bincount(y[indices], minlength=2),
                               [len(indices) // 2, len(indices) // 2]),
                f"selection {subset} split is not stratified as specified.")
        data[f"{subset}_indices"] = indices.copy()
        data[f"X_{subset}"] = X[indices].copy()
        label_key = "y_test" if subset == "test" else f"y_{subset}_clean"
        data[label_key] = y[indices].copy()
    for array in data.values():
        array.setflags(write=False)
    return data


def make_noisy_label_conditions(y_clean, rates, flip_seed, subset_name):
    """One permutation per subset; cache all levels to reuse across conditions."""
    permutation = np.random.default_rng(flip_seed).permutation(len(y_clean))
    replay_permutation = np.random.default_rng(flip_seed).permutation(len(y_clean))
    original = y_clean.copy()
    conditions = {}
    previous_flips = set()
    for rate in rates:
        labels, indices, actual_rate = flip_binary_labels(y_clean, rate, permutation, subset_name)
        replay_labels, replay_indices, replay_rate = flip_binary_labels(
            y_clean, rate, replay_permutation, subset_name,
        )
        require(np.array_equal(labels, replay_labels) and np.array_equal(indices, replay_indices)
                and actual_rate == replay_rate, f"{subset_name} label flips are not reproducible.")
        current_flips = set(indices.tolist())
        require(previous_flips.issubset(current_flips), f"{subset_name} flip sets are not nested.")
        previous_flips = current_flips
        indices.setflags(write=False)
        conditions[rate] = {"labels": labels, "indices": indices, "actual_rate": actual_rate}
    require(np.array_equal(y_clean, original), f"original {subset_name} labels changed.")
    return conditions


def select_depth(validation_predictions, y_observed):
    """Select only by observed validation accuracy; there is no test-data argument."""
    require(set(validation_predictions) == set(SELECTION_TIE_ORDER),
            "selection needs validation predictions for all four candidate depths.")
    require(y_observed.ndim == 1 and len(y_observed) > 0 and np.isin(y_observed, [0, 1]).all(),
            "observed validation labels must be a nonempty binary array.")
    correct_counts = []
    for label in SELECTION_TIE_ORDER:
        predictions = validation_predictions[label]
        require(predictions.shape == y_observed.shape and np.isin(predictions, [0, 1]).all(),
                f"invalid validation predictions for depth {label}.")
        correct_counts.append(int(np.count_nonzero(predictions == y_observed)))
    # Integer counts avoid approximate floating-point tie decisions. np.argmax
    # returns the FIRST maximum, so the pre-specified order determines exact ties.
    winner = int(np.argmax(correct_counts))
    return SELECTION_TIE_ORDER[winner], correct_counts[winner] / len(y_observed)


def validate_selection_rule():
    """Small deterministic examples verify both score maximization and exact ties."""
    require(tuple(signature(select_depth).parameters) == ("validation_predictions", "y_observed"),
            "selection must accept only validation predictions and observed labels.")
    y = np.array([0, 0, 1, 1])
    opposite = 1 - y
    # Deliberately reverse dictionary insertion order: the tie rule must still win.
    predictions = {label: y.copy() for label in reversed(SELECTION_TIE_ORDER)}
    require(select_depth(predictions, y) == ("2", 1.0), "all-way tie must select depth 2.")
    predictions["2"] = opposite.copy()
    predictions["unlimited"] = opposite.copy()
    require(select_depth(predictions, y) == ("4", 1.0), "tie between 4 and 8 must select 4.")
    require(select_depth(predictions, opposite) == ("2", 1.0),
            "selection must follow observed labels when those labels change.")
    predictions["4"] = opposite.copy()
    predictions["8"] = opposite.copy()
    predictions["unlimited"] = y.copy()
    require(select_depth(predictions, y) == ("unlimited", 1.0),
            "a unique highest score must win even for unlimited depth.")
    print("Validation passed: selection uses observed labels; deterministic tie-rule examples passed.")


def fit_selection_candidates(data, y_train_observed, model_seed, reference_data, context):
    """Fit four trees on training data only; retain models and validation predictions."""
    candidates, predictions = {}, {}
    reference_y = y_train_observed.copy()
    fit_count = 0
    for max_depth in SELECTION_DEPTHS:
        label = depth_label(max_depth)
        validate_condition_inputs(data, reference_data, y_train_observed, reference_y, context)
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=model_seed, **TREE_SETTINGS)
        model.fit(data["X_train"], y_train_observed)
        fit_count += 1
        require(max_depth is None or model.get_depth() <= max_depth,
                f"{context}: candidate exceeds its depth limit.")
        candidates[label] = model
        predictions[label] = model.predict(data["X_validation"])
        predictions[label].setflags(write=False)
        validate_condition_inputs(data, reference_data, y_train_observed, reference_y, context)
    return candidates, predictions, fit_count


def evaluate_recorded_selections(selections, candidates, X_test, y_test):
    """Only after ALL four choices are recorded may the untouched test set be used."""
    require(set(selections) == set(SELECTION_VALIDATION_NOISE_RATES),
            "all validation-noise selections must be recorded before test evaluation.")
    require(all("selected_depth" in choice for choice in selections.values()),
            "test evaluation received an incomplete selection record.")
    # Evaluate only models actually selected, once per distinct selected depth.
    chosen_depths = {choice["selected_depth"] for choice in selections.values()}
    return {label: accuracy_score(y_test, candidates[label].predict(X_test))
            for label in SELECTION_TIE_ORDER if label in chosen_depths}


def run_selection_repetitions():
    """480 candidate fits produce 480 outcomes; validation noise never triggers a refit."""
    rows, fit_count = [], 0
    for repetition_number, seed in enumerate(SELECTION_SEEDS, start=1):
        seeds = make_selection_random_seeds(seed)
        data = generate_selection_data(seeds)
        reference_data = {name: array.copy() for name, array in data.items()}
        replay = generate_selection_data(make_selection_random_seeds(seed))
        require(all(np.array_equal(data[name], replay[name]) for name in data),
                f"selection seed {seed}: generated split is not reproducible.")
        training_conditions = make_noisy_label_conditions(
            data["y_train_clean"], SELECTION_TRAIN_NOISE_RATES, seeds["train_flips"], "training",
        )
        # Construct these ONCE per repetition, outside the training-noise loop.
        validation_conditions = make_noisy_label_conditions(
            data["y_validation_clean"], SELECTION_VALIDATION_NOISE_RATES,
            seeds["validation_flips"], "validation",
        )
        reference_validation_labels = {
            rate: condition["labels"].copy() for rate, condition in validation_conditions.items()
        }
        for train_rate, train_condition in training_conditions.items():
            context = f"selection seed {seed}, training noise {train_rate}"
            train_labels = train_condition["labels"]
            reference_train_labels = train_labels.copy()
            candidates, predictions, condition_fits = fit_selection_candidates(
                data, train_labels, seeds["model"], reference_data, context,
            )
            fit_count += condition_fits
            prediction_snapshot = {label: values.copy() for label, values in predictions.items()}
            selections = {}
            for validation_rate, validation_condition in validation_conditions.items():
                observed_labels = validation_condition["labels"]
                require(np.array_equal(observed_labels, reference_validation_labels[validation_rate]),
                        f"{context}: validation flip labels changed between training-noise conditions.")
                label, observed_accuracy = select_depth(predictions, observed_labels)
                # Independent check: find all highest-scoring candidates in the fixed
                # order and verify that the selected one is the first of those ties.
                scores = [accuracy_score(observed_labels, predictions[d]) for d in SELECTION_TIE_ORDER]
                expected_label = next(d for d, score in zip(SELECTION_TIE_ORDER, scores)
                                      if score == max(scores))
                require(label == expected_label and observed_accuracy == max(scores),
                        f"{context}: selection disagrees with observed accuracy or the tie rule.")
                selections[validation_rate] = {
                    "selected_depth": label, "selected_validation_accuracy_observed": observed_accuracy,
                }
                for candidate_label, model in candidates.items():
                    require(np.array_equal(predictions[candidate_label], prediction_snapshot[candidate_label])
                            and np.array_equal(model.predict(data["X_validation"]),
                                               prediction_snapshot[candidate_label]),
                            f"{context}: candidate predictions changed with validation noise.")
                validate_condition_inputs(data, reference_data, train_labels, reference_train_labels, context)

            # No test predictions or metrics have been computed above. All choices
            # are now final; do not retrain on training + validation samples.
            test_accuracies = evaluate_recorded_selections(
                selections, candidates, data["X_test"], data["y_test"],
            )
            reference_depth = selections[0.0]["selected_depth"]
            reference_test_accuracy = test_accuracies[reference_depth]
            for validation_rate, choice in selections.items():
                validation_condition = validation_conditions[validation_rate]
                label = choice["selected_depth"]
                selected_test_accuracy = test_accuracies[label]
                rows.append({
                    "seed": seed, "requested_train_noise_rate": train_rate,
                    "actual_train_noise_rate": train_condition["actual_rate"],
                    "requested_validation_noise_rate": validation_rate,
                    "actual_validation_noise_rate": validation_condition["actual_rate"],
                    "n_train_flipped": len(train_condition["indices"]),
                    "n_validation_flipped": len(validation_condition["indices"]),
                    "n_train": len(train_labels), "n_validation": len(data["y_validation_clean"]),
                    "n_test": len(data["y_test"]), **choice,
                    # Original validation labels are a diagnostic, never an input
                    # to selection under a nonzero validation-noise condition.
                    "selected_validation_accuracy_clean": accuracy_score(
                        data["y_validation_clean"], predictions[label]),
                    "selected_test_accuracy": selected_test_accuracy,
                    "reference_depth_clean_validation": reference_depth,
                    "reference_test_accuracy_clean_validation": reference_test_accuracy,
                    SELECTION_CHANGED: label != reference_depth,
                    SELECTION_CHANGE: selected_test_accuracy - reference_test_accuracy,
                })
            validate_condition_inputs(data, reference_data, train_labels, reference_train_labels, context)
        print(f"Selection repetition {repetition_number:02d}/30 (seed={seed:02d}): "
              "16 candidate fits, 16 choices; data, flips, and selection checks passed.", flush=True)
    require(fit_count == SELECTION_EXPECTED_FITS,
            f"expected 480 new candidate fits, got {fit_count}.")
    require(len(rows) == SELECTION_EXPECTED_OUTCOMES, "expected 480 selection outcomes.")
    return pd.DataFrame(rows, columns=SELECTION_RAW_COLUMNS)


def summarize_selection_results(raw):
    """Summarize already-paired changes and include every depth, even when never chosen."""
    summary_rows, frequency_rows = [], []
    for train_rate, validation_rate in product(SELECTION_TRAIN_NOISE_RATES,
                                                SELECTION_VALIDATION_NOISE_RATES):
        group = raw.loc[(raw["requested_train_noise_rate"] == train_rate)
                        & (raw["requested_validation_noise_rate"] == validation_rate)]
        keys = {"requested_train_noise_rate": train_rate,
                "requested_validation_noise_rate": validation_rate}
        row = {**keys, "repetition_count": group["seed"].nunique(),
               "selection_changed_proportion": group[SELECTION_CHANGED].mean()}
        for metric in ("selected_test_accuracy", SELECTION_CHANGE):
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        summary_rows.append(row)
        for label in SELECTION_TIE_ORDER:
            count = int((group["selected_depth"] == label).sum())
            frequency_rows.append({**keys, "selected_depth": label,
                                   "selection_count": count, "selection_proportion": count / len(group)})
    return pd.DataFrame(summary_rows), pd.DataFrame(frequency_rows)


def validate_selection_results(raw, summary=None, frequencies=None, announce=True):
    """Check every new outcome and its matched reference, then verify both summaries."""
    require(tuple(raw.columns) == SELECTION_RAW_COLUMNS, "selection raw columns do not match the design.")
    keys = ["seed", *SELECTION_GROUP_KEYS]
    require(len(raw) == SELECTION_EXPECTED_OUTCOMES and not raw.duplicated(keys).any()
            and set(raw[keys].itertuples(index=False, name=None)) == set(product(
                SELECTION_SEEDS, SELECTION_TRAIN_NOISE_RATES, SELECTION_VALIDATION_NOISE_RATES)),
            "selection raw results must contain exactly 480 unique pre-specified outcomes.")
    require(not raw.isna().any().any(), "selection raw results contain missing values.")
    numeric_columns = [name for name in SELECTION_RAW_COLUMNS
                       if name not in ("selected_depth", "reference_depth_clean_validation", SELECTION_CHANGED)]
    require(np.isfinite(raw[numeric_columns].to_numpy(dtype=float)).all(),
            "selection raw numeric results contain missing or infinite values.")
    for name in SELECTION_ACCURACIES:
        require(raw[name].between(0.0, 1.0).all(), f"selection {name} is outside [0, 1].")
    require(raw[SELECTION_CHANGE].between(-1.0, 1.0).all(), "test-accuracy changes are outside [-1, 1].")
    require(pd.api.types.is_bool_dtype(raw[SELECTION_CHANGED]), "selection-changed flags must be booleans.")
    for name in ("selected_depth", "reference_depth_clean_validation"):
        require(raw[name].isin(SELECTION_TIE_ORDER).all(), f"unknown depth in {name}.")
    for subset, size in (("train", 600), ("validation", 200), ("test", 200)):
        require((raw[f"n_{subset}"] == size).all(), f"incorrect recorded selection {subset} size.")
    for subset in ("train", "validation"):
        expected_counts = [round(rate * size) for rate, size in
                           zip(raw[f"requested_{subset}_noise_rate"], raw[f"n_{subset}"])]
        require(np.array_equal(raw[f"n_{subset}_flipped"], expected_counts),
                f"incorrect recorded {subset} flip counts.")
        require(np.allclose(raw[f"actual_{subset}_noise_rate"],
                            raw[f"n_{subset}_flipped"] / raw[f"n_{subset}"],
                            rtol=0, atol=NUMERIC_TOLERANCE), f"incorrect actual {subset} flip rates.")
    reference_rows = raw.loc[raw["requested_validation_noise_rate"] == 0.0]
    references = reference_rows.set_index(["seed", "requested_train_noise_rate"])
    for row in raw.to_dict("records"):
        reference = references.loc[(row["seed"], row["requested_train_noise_rate"])]
        require(row["reference_depth_clean_validation"] == reference["selected_depth"],
                "reference depth does not match the same seed and training-noise condition.")
        require(np.isclose(row["reference_test_accuracy_clean_validation"],
                           reference["selected_test_accuracy"], rtol=0, atol=NUMERIC_TOLERANCE),
                "reference test accuracy is matched to the wrong condition.")
        require(row[SELECTION_CHANGED] == (row["selected_depth"] != reference["selected_depth"]),
                "selection-changed flag disagrees with the matched depths.")
        expected_change = row["selected_test_accuracy"] - reference["selected_test_accuracy"]
        require(np.isclose(row[SELECTION_CHANGE], expected_change, rtol=0, atol=NUMERIC_TOLERANCE),
                "paired test-accuracy change does not match the same seed and training-noise reference.")
        if not row[SELECTION_CHANGED]:
            require(row[SELECTION_CHANGE] == 0.0, "unchanged selected depth has different test accuracy.")
    require(not reference_rows[SELECTION_CHANGED].any()
            and (reference_rows[SELECTION_CHANGE] == 0.0).all(),
            "zero validation noise must give zero paired change and no selection changes.")
    require(np.allclose(reference_rows["selected_validation_accuracy_observed"],
                        reference_rows["selected_validation_accuracy_clean"], rtol=0, atol=NUMERIC_TOLERANCE),
            "observed and clean validation scores differ at zero validation noise.")
    for _, group in raw.groupby(["seed", "requested_train_noise_rate", "selected_depth"]):
        require(group["selected_test_accuracy"].nunique() == 1
                and group["selected_validation_accuracy_clean"].nunique() == 1,
                "the same candidate has inconsistent diagnostic scores across validation-noise conditions.")
    if summary is None and frequencies is None:
        if announce:
            print("Validation passed: 480 selection outcomes, valid metrics, and correctly paired references.")
        return
    require(summary is not None and frequencies is not None, "both selection summaries are required.")
    group_keys = list(SELECTION_GROUP_KEYS)
    expected_groups = set(product(SELECTION_TRAIN_NOISE_RATES, SELECTION_VALIDATION_NOISE_RATES))
    require(len(summary) == SELECTION_EXPECTED_GROUPS and not summary.duplicated(group_keys).any()
            and set(summary[group_keys].itertuples(index=False, name=None)) == expected_groups
            and (summary["repetition_count"] == 30).all(),
            "selection summary must contain 16 groups of 30 repetitions.")
    require(np.isfinite(summary.to_numpy(dtype=float)).all(), "selection summary contains nonfinite values.")
    require(summary["selected_test_accuracy_mean"].between(0.0, 1.0).all()
            and summary["selection_changed_proportion"].between(0.0, 1.0).all()
            and summary[f"{SELECTION_CHANGE}_mean"].between(-1.0, 1.0).all(),
            "selection summary means or proportions have invalid ranges.")
    frequency_keys = [*group_keys, "selected_depth"]
    require(len(frequencies) == SELECTION_EXPECTED_FREQUENCY_ROWS
            and not frequencies.duplicated(frequency_keys).any()
            and set(frequencies[frequency_keys].itertuples(index=False, name=None)) == set(product(
                SELECTION_TRAIN_NOISE_RATES, SELECTION_VALIDATION_NOISE_RATES, SELECTION_TIE_ORDER)),
            "selection frequencies must include all 64 categories, including zero-count depths.")
    require(np.isfinite(frequencies.drop(columns="selected_depth").to_numpy(dtype=float)).all(),
            "selection frequencies contain missing or infinite numeric values.")
    require(frequencies["selection_count"].between(0, 30).all()
            and frequencies["selection_proportion"].between(0.0, 1.0).all(),
            "selection frequency counts or proportions have invalid ranges.")
    for row in summary.to_dict("records"):
        train_rate, validation_rate = (row[key] for key in group_keys)
        group = raw.loc[(raw[group_keys[0]] == train_rate) & (raw[group_keys[1]] == validation_rate)]
        require(len(group) == 30 and set(group["seed"]) == set(SELECTION_SEEDS),
                "a selection summary group is missing or duplicates repetition seeds.")
        for metric in ("selected_test_accuracy", SELECTION_CHANGE):
            values = group[metric].to_numpy()
            for suffix, expected in (("mean", np.mean(values)), ("std", np.std(values, ddof=1))):
                require(np.isclose(row[f"{metric}_{suffix}"], expected, rtol=0, atol=NUMERIC_TOLERANCE),
                        f"incorrect selection {metric}_{suffix}; sample SD must use ddof=1.")
        require(np.isclose(row["selection_changed_proportion"], group[SELECTION_CHANGED].mean(),
                           rtol=0, atol=NUMERIC_TOLERANCE), "incorrect proportion of changed selections.")
        frequency_group = frequencies.loc[(frequencies[group_keys[0]] == train_rate)
                                          & (frequencies[group_keys[1]] == validation_rate)]
        require(frequency_group["selection_count"].sum() == 30
                and np.isclose(frequency_group["selection_proportion"].sum(), 1.0,
                               rtol=0, atol=NUMERIC_TOLERANCE),
                "each depth-frequency group must total 30 selections and proportions must sum to 1.")
        for frequency in frequency_group.to_dict("records"):
            expected_count = int((group["selected_depth"] == frequency["selected_depth"]).sum())
            require(frequency["selection_count"] == expected_count
                    and np.isclose(frequency["selection_proportion"], expected_count / 30,
                                   rtol=0, atol=NUMERIC_TOLERANCE), "depth frequencies disagree with raw selections.")
    if announce:
        print("Validation passed: 480 outcomes; 16 paired summary groups; 64 depth-frequency rows; "
              "sample SDs and reference comparisons verified.")


def create_figure_4(summary):
    """Performance effects of validation noise: preserve negative and positive changes."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8), sharex=True, sharey=True)
    for ax, train_rate in zip(axes.flat, SELECTION_TRAIN_NOISE_RATES):
        group = summary.loc[summary["requested_train_noise_rate"] == train_rate].sort_values(
            "requested_validation_noise_rate")
        ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1)
        ax.errorbar(group["requested_validation_noise_rate"], 100 * group[f"{SELECTION_CHANGE}_mean"],
                    yerr=100 * group[f"{SELECTION_CHANGE}_std"], color="#0072B2",
                    marker="o", linewidth=2, capsize=4, label="Mean ± one sample SD")
        ax.set_title(f"Training labels flipped: {train_rate:.0%}")
        ax.set_xticks(SELECTION_VALIDATION_NOISE_RATES)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.grid(True, alpha=0.20)
    # Shared autoscaling includes every mean +/- SD; no monotonicity is imposed.
    fig.suptitle("Effect of Validation Label Noise on Selected-Model Performance", fontsize=15, y=0.97)
    fig.supxlabel("Validation labels flipped (%)", y=0.145)
    fig.supylabel("Mean paired change in test accuracy (percentage points)", x=0.015, y=0.57)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.09), frameon=False)
    fig.text(0.5, 0.04, "30 simulations per panel; error bars show sample SD, not a confidence interval.\n"
             "Zero = clean-validation selection; negative = worse test accuracy; positive = better.",
             ha="center", va="center", fontsize=10)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.89, bottom=0.23, hspace=0.28, wspace=0.18)
    fig.savefig(SELECTION_FIGURE_PATHS[0], dpi=FIGURE_DPI)
    plt.close(fig)


def create_figure_5(frequencies):
    """Selection frequencies answer a different question from performance changes."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8), sharex=True, sharey=True)
    for ax, train_rate in zip(axes.flat, SELECTION_TRAIN_NOISE_RATES):
        bottom = np.zeros(len(SELECTION_VALIDATION_NOISE_RATES))
        for label in SELECTION_TIE_ORDER:
            group = frequencies.loc[(frequencies["requested_train_noise_rate"] == train_rate)
                                    & (frequencies["selected_depth"] == label)].sort_values(
                "requested_validation_noise_rate")
            proportions = group["selection_proportion"].to_numpy()
            ax.bar(SELECTION_VALIDATION_NOISE_RATES, proportions, bottom=bottom, width=0.06,
                   color=DEPTH_COLORS[label], edgecolor="white", linewidth=0.7, label=f"Depth {label}")
            bottom += proportions
        require(np.allclose(bottom, 1.0, rtol=0, atol=NUMERIC_TOLERANCE),
                "Figure 5 bars must total 100%.")
        ax.set_title(f"Training labels flipped: {train_rate:.0%}")
        ax.set_xticks(SELECTION_VALIDATION_NOISE_RATES)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.20)
    fig.suptitle("Selected Tree Depth under Validation Label Noise", fontsize=16, y=0.97)
    fig.supxlabel("Validation labels flipped (%)", y=0.145)
    fig.supylabel("Selection proportion (%)", x=0.015, y=0.57)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.09), ncols=4, frameon=False)
    fig.text(0.5, 0.04, "Each bar summarizes 30 repetitions and totals 100%. Exact ties prefer 2, then 4, then 8, then unlimited.\n"
             "A change in selected depth does not necessarily reduce test accuracy.",
             ha="center", va="center", fontsize=10)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.89, bottom=0.23, hspace=0.28, wspace=0.18)
    fig.savefig(SELECTION_FIGURE_PATHS[1], dpi=FIGURE_DPI)
    plt.close(fig)


def validate_selection_output_files():
    for path in (SELECTION_RAW_PATH, SELECTION_SUMMARY_PATH, SELECTION_FREQUENCIES_PATH,
                 *SELECTION_FIGURE_PATHS):
        require(path.is_file() and path.stat().st_size > 0,
                f"required selection output is missing or empty: {path}")
    raw = pd.read_csv(SELECTION_RAW_PATH, dtype={"selected_depth": str, "reference_depth_clean_validation": str})
    summary = pd.read_csv(SELECTION_SUMMARY_PATH)
    frequencies = pd.read_csv(SELECTION_FREQUENCIES_PATH, dtype={"selected_depth": str})
    validate_selection_results(raw, summary, frequencies, announce=False)
    print("Validation passed: three selection CSVs pass checks after reloading; both new PNGs are nonempty.")


def run_validation_noise_experiment():
    print("\nCan Tree Depth Be Selected Reliably When Validation Labels Are Also Noisy?", flush=True)
    require(SELECTION_N_SAMPLES == 1000 and SELECTION_FEATURE_NOISE == 0.20
            and SELECTION_SEEDS == tuple(range(30))
            and SELECTION_HOLDOUT_FRACTION == 0.40 and SELECTION_TEST_SHARE_OF_HOLDOUT == 0.50
            and SELECTION_TRAIN_NOISE_RATES == (0.0, 0.1, 0.2, 0.3)
            and SELECTION_VALIDATION_NOISE_RATES == (0.0, 0.1, 0.2, 0.3)
            and SELECTION_DEPTHS == (2, 4, 8, None) and SELECTION_TIE_ORDER == ("2", "4", "8", "unlimited"),
            "the pre-specified validation-noise experiment design changed.")
    print("Separate 600/200/200 split; 480 new candidate fits and 480 selection outcomes.\n"
          "All four choices are recorded before test evaluation; no training + validation refits.", flush=True)
    validate_selection_rule()
    SELECTION_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    raw = run_selection_repetitions()
    validate_selection_results(raw, announce=False)
    summary, frequencies = summarize_selection_results(raw)
    validate_selection_results(raw, summary, frequencies)
    raw.to_csv(SELECTION_RAW_PATH, index=False)
    summary.to_csv(SELECTION_SUMMARY_PATH, index=False)
    frequencies.to_csv(SELECTION_FREQUENCIES_PATH, index=False)
    create_figure_4(summary)
    create_figure_5(frequencies)
    validate_selection_output_files()
    print("\nValidation noise = 30%: selected-model test accuracy, paired change, and changed selections")
    for row in summary.loc[summary["requested_validation_noise_rate"] == 0.30].to_dict("records"):
        print(f"  Training flips {row['requested_train_noise_rate']:.0%}: "
              f"test {row['selected_test_accuracy_mean']:.2%}; "
              f"change {100 * row[SELECTION_CHANGE + '_mean']:+.2f} ± "
              f"{100 * row[SELECTION_CHANGE + '_std']:.2f} percentage points (mean ± SD); "
              f"depth changed {row['selection_changed_proportion']:.0%}")
    print("SD describes simulation variability, not a confidence interval. Clean validation is a reference,\n"
          "not an oracle. Positive changes are retained. Do not pool these results with Experiment 1.")
    print("\nValidation-noise output locations:")
    for path in (SELECTION_RAW_PATH, SELECTION_SUMMARY_PATH, SELECTION_FREQUENCIES_PATH,
                 *SELECTION_FIGURE_PATHS):
        print(f"  {path}")


def plot_paired_repetitions(ax, values_by_depth):
    """Show every repetition, plus a mean and sample SD, in percentage points."""
    for position, (label, values) in enumerate(values_by_depth.items()):
        values = np.asarray(values, dtype=float)
        require(len(values) == len(REPETITION_SEEDS) and np.isfinite(values).all(),
                "additional paired plot needs 20 finite values per depth.")
        # Fixed horizontal spacing prevents overlap; it does not change the values
        # or introduce another random-number stream. Input rows are sorted by seed.
        offsets = np.linspace(-0.22, 0.04, len(values))
        ax.scatter(position + offsets, 100 * values, s=20, alpha=0.60,
                   color=DEPTH_COLORS[label], edgecolors="none", zorder=3)
        ax.errorbar(position + 0.18, 100 * values.mean(),
                    yerr=100 * values.std(ddof=1), fmt="D", markersize=6,
                    color=DEPTH_COLORS[label], markeredgecolor="#333333",
                    capsize=5, linewidth=1.8, zorder=4)
    ax.axhline(0, color="#555555", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(values_by_depth)), list(values_by_depth))
    for tick, label in zip(ax.get_xticklabels(), values_by_depth):
        tick.set_color(DEPTH_COLORS[label])
    ax.set_xlim(-0.5, len(values_by_depth) - 0.5)
    ax.grid(axis="y", alpha=0.20)
    ax.set_axisbelow(True)


def create_figure_6(raw):
    """Compare each limited tree with the unlimited tree in the SAME simulation."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for ax, rate in zip(axes.flat, NOISE_RATES):
        group = raw.loc[raw["requested_noise_rate"] == rate].sort_values("seed")
        values = {label: group.loc[group["max_depth"] == label, PAIRED_COLUMN].to_numpy()
                  for label in DEPTH_LABELS[:-1]}
        plot_paired_repetitions(ax, values)
        ax.set_title(f"Training labels flipped: {rate:.0%}")
        ax.tick_params(labelleft=True)
    fig.suptitle("Test Accuracy: Limited Depth Minus Unlimited Depth", fontsize=16, y=0.97)
    fig.supxlabel("Maximum tree depth", y=0.13)
    fig.supylabel("Paired test-accuracy difference (percentage points)", x=0.02, y=0.57)
    fig.text(0.5, 0.055, "Small dots: all 20 simulations. Diamonds and bars: mean ± one sample SD (not a CI).\n"
             "Paired within the same seed and training-noise level. Positive = better than unlimited.",
             ha="center", va="center", fontsize=10)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.89, bottom=0.22, hspace=0.28, wspace=0.15)
    fig.savefig(ADDITIONAL_TRAIN_FIGURE_PATHS[0], dpi=FIGURE_DPI)
    plt.close(fig)


def create_figure_7(raw):
    """Pair noisy-training test scores with the same depth at zero training flips."""
    reference = raw.loc[raw["requested_noise_rate"] == 0,
                        ["seed", "max_depth", "test_accuracy"]].rename(
                            columns={"test_accuracy": "reference_test_accuracy"})
    paired = raw.merge(reference, on=["seed", "max_depth"], validate="many_to_one")
    paired["change"] = paired["test_accuracy"] - paired["reference_test_accuracy"]
    require(len(paired) == EXPECTED_FITS and not paired["change"].isna().any(),
            "Figure 7 could not match every seed and depth to its zero-noise reference.")
    require((paired.loc[paired["requested_noise_rate"] == 0, "change"] == 0).all(),
            "Figure 7 zero-noise comparisons must be zero.")
    fig, axes = plt.subplots(1, 3, figsize=(13, 6.2), sharey=True)
    for ax, rate in zip(axes, NOISE_RATES[1:]):
        group = paired.loc[paired["requested_noise_rate"] == rate].sort_values("seed")
        values = {label: group.loc[group["max_depth"] == label, "change"].to_numpy()
                  for label in DEPTH_LABELS}
        plot_paired_repetitions(ax, values)
        ax.set_title(f"Training labels flipped: {rate:.0%}")
        ax.tick_params(labelleft=True)
    fig.suptitle("How Much Does Test Accuracy Change When Training Labels Are Flipped?",
                 fontsize=15, y=0.97)
    fig.supxlabel("Maximum tree depth", y=0.17)
    fig.supylabel("Change from 0% training flips (percentage points)", x=0.015, y=0.59)
    fig.text(0.5, 0.07, "Each point uses the same seed and depth as its 0%-flip reference; negative = lower test accuracy.\n"
             "Small dots: 20 simulations. Diamonds and bars: mean ± sample SD (not a CI).\n"
             "Losing less accuracy does not necessarily mean having higher final accuracy.",
             ha="center", va="center", fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.28, wspace=0.18)
    fig.savefig(ADDITIONAL_TRAIN_FIGURE_PATHS[1], dpi=FIGURE_DPI)
    plt.close(fig)


def create_figure_8(raw):
    """Measure fitted complexity directly, rather than treating depth as accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.5), sharex=True)
    for ax, metric, title in zip(axes, ("fitted_tree_depth", "n_leaves"),
                                ("Actual fitted depth", "Number of leaves")):
        for label, marker in zip(DEPTH_LABELS, ("o", "s", "^", "D")):
            group = raw.loc[raw["max_depth"] == label].groupby("requested_noise_rate")[metric]
            means = group.mean().reindex(NOISE_RATES)
            spreads = group.std(ddof=1).reindex(NOISE_RATES)
            ax.errorbar(NOISE_RATES, means, yerr=spreads, color=DEPTH_COLORS[label],
                        marker=marker, linewidth=2, capsize=4, label=f"Depth {label}")
        ax.set_title(title)
        ax.set_ylabel(f"Mean {title.lower()}")
        ax.set_xticks(NOISE_RATES)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.20)
    fig.suptitle("Fitted Tree Complexity under Training Label Noise", fontsize=16, y=0.97)
    fig.supxlabel("Training labels flipped (%)", y=0.21)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.115),
               ncols=4, frameon=False)
    fig.text(0.5, 0.055, "Means ± one sample SD across 20 simulations; SD is not a confidence interval.\n"
             "Depth counts splitting steps on the longest path. Leaves are the final prediction regions.",
             ha="center", va="center", fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.87, bottom=0.32, wspace=0.23)
    fig.savefig(ADDITIONAL_TRAIN_FIGURE_PATHS[2], dpi=FIGURE_DPI)
    plt.close(fig)


def create_figure_9(summary):
    """Make the unflipped-validation reference visible as an absolute test score."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), sharex=True, sharey=True)
    for ax, rate in zip(axes.flat, SELECTION_TRAIN_NOISE_RATES):
        group = summary.loc[summary["requested_train_noise_rate"] == rate].sort_values(
            "requested_validation_noise_rate")
        reference = group.loc[group["requested_validation_noise_rate"] == 0,
                              "selected_test_accuracy_mean"].item()
        ax.axhline(reference, linestyle="--", color="#555555", linewidth=1.4,
                   label="0% validation flips: reference mean")
        ax.errorbar(group["requested_validation_noise_rate"], group["selected_test_accuracy_mean"],
                    yerr=group["selected_test_accuracy_std"], marker="o", capsize=4,
                    color="#0072B2", linewidth=2, label="Current selection: mean ± SD")
        ax.text(0.03, 0.06, f"Reference mean: {reference:.2%}", transform=ax.transAxes,
                fontsize=10, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
        ax.set_title(f"Training labels flipped: {rate:.0%}")
        ax.set_xticks(SELECTION_VALIDATION_NOISE_RATES)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.grid(alpha=0.20)
    fig.suptitle("Selected-Model Test Accuracy with the Clean-Validation Reference", fontsize=15, y=0.97)
    fig.supxlabel("Validation labels flipped (%)", y=0.18)
    fig.supylabel("Mean test accuracy (%)", x=0.015, y=0.59)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.115),
               ncols=2, frameon=False, fontsize=10)
    fig.text(0.5, 0.05, "30 simulations; bars show sample SD, not a confidence interval. Test labels are untouched.\n"
             "The dashed line is the mean test score after selection using original validation labels.\n"
             "It is a controlled reference, not an oracle. Figure 4 shows the paired differences.",
             ha="center", va="center", fontsize=10)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.89, bottom=0.27, hspace=0.30, wspace=0.15)
    fig.savefig(ADDITIONAL_SELECTION_FIGURE_PATHS[0], dpi=FIGURE_DPI)
    plt.close(fig)


def create_figure_10(raw):
    """Count depth-to-depth transitions at the highest studied validation flip rate."""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.5))
    for ax, rate in zip(axes.flat, SELECTION_TRAIN_NOISE_RATES):
        group = raw.loc[(raw["requested_train_noise_rate"] == rate)
                        & (raw["requested_validation_noise_rate"] == TRANSITION_VALIDATION_RATE)]
        counts = pd.crosstab(group["reference_depth_clean_validation"], group["selected_depth"])
        counts = counts.reindex(index=DEPTH_LABELS, columns=DEPTH_LABELS, fill_value=0).to_numpy()
        require(counts.sum() == len(SELECTION_SEEDS)
                and counts.sum() - np.trace(counts) == group[SELECTION_CHANGED].sum(),
                "Figure 10 transition counts disagree with selection-change flags.")
        plot = ax.imshow(counts, cmap="Blues", vmin=0, vmax=len(SELECTION_SEEDS))
        for row, column in product(range(4), repeat=2):
            count = int(counts[row, column])
            ax.text(column, row, str(count), ha="center", va="center", fontsize=12,
                    color="white" if count > len(SELECTION_SEEDS) / 2 else "#333333")
        ax.set_xticks(range(4), DEPTH_LABELS)
        ax.set_yticks(range(4), DEPTH_LABELS)
        for ticks in (ax.get_xticklabels(), ax.get_yticklabels()):
            for tick, label in zip(ticks, DEPTH_LABELS):
                tick.set_color(DEPTH_COLORS[label])
        ax.set_title(f"Training labels flipped: {rate:.0%}", pad=10)
    color_ax = fig.add_axes([0.89, 0.28, 0.022, 0.50])
    fig.colorbar(plot, cax=color_ax, ticks=range(0, len(SELECTION_SEEDS) + 1, 5), label="Number of repetitions")
    fig.suptitle("Which Tree Depth Was Chosen After Validation Labels Were Flipped?",
                 fontsize=14, y=0.97)
    fig.supxlabel(f"Selected depth with {TRANSITION_VALIDATION_RATE:.0%} validation flips", y=0.14)
    fig.supylabel("Selected depth with 0% validation flips", x=0.02, y=0.56)
    fig.text(0.5, 0.06, "Each panel counts all 30 matched simulations. Rows = original choice; columns = new choice.\n"
             "Diagonal cells: same depth. Other cells: changed depth. Counts do not measure accuracy.\n"
             "Only the highest studied validation-noise level is shown here; Figure 5 includes all levels.",
             ha="center", va="center", fontsize=10)
    fig.subplots_adjust(left=0.11, right=0.85, top=0.89, bottom=0.22, hspace=0.30, wspace=0.30)
    fig.savefig(ADDITIONAL_SELECTION_FIGURE_PATHS[1], dpi=FIGURE_DPI)
    plt.close(fig)


def create_figure_11(raw):
    """Changing depth and changing accuracy are different; keep improved outcomes."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8), sharex=True, sharey=True)
    names = ("Lower test accuracy", "Same test accuracy", "Higher test accuracy")
    for ax, rate in zip(axes.flat, SELECTION_TRAIN_NOISE_RATES):
        counts = []
        for validation_rate in SELECTION_VALIDATION_NOISE_RATES:
            group = raw.loc[(raw["requested_train_noise_rate"] == rate)
                            & (raw["requested_validation_noise_rate"] == validation_rate)]
            changes = group[SELECTION_CHANGE].to_numpy()
            # Tolerance handles CSV floating-point rounding only; a single test
            # prediction changes accuracy by 1/200, far larger than this tolerance.
            counts.append([int((changes < -NUMERIC_TOLERANCE).sum()),
                           int((np.abs(changes) <= NUMERIC_TOLERANCE).sum()),
                           int((changes > NUMERIC_TOLERANCE).sum())])
        counts = np.asarray(counts)
        require((counts.sum(axis=1) == len(SELECTION_SEEDS)).all()
                and np.array_equal(counts[0], [0, len(SELECTION_SEEDS), 0]),
                "Figure 11 must count 30 outcomes per bar, all unchanged at zero validation noise.")
        bottom = np.zeros(len(SELECTION_VALIDATION_NOISE_RATES))
        for index, (name, color) in enumerate(zip(names, PERFORMANCE_OUTCOME_COLORS)):
            heights = counts[:, index] / len(SELECTION_SEEDS)
            ax.bar(SELECTION_VALIDATION_NOISE_RATES, heights, bottom=bottom, width=0.06,
                   label=name, color=color, edgecolor="white", linewidth=0.7)
            for x, count, height, base in zip(SELECTION_VALIDATION_NOISE_RATES,
                                              counts[:, index], heights, bottom):
                if count:
                    ax.text(x, base + height / 2, str(count), ha="center", va="center",
                            fontsize=10, color="#222222" if index == 1 else "white")
            bottom += heights
        ax.set_title(f"Training labels flipped: {rate:.0%}")
        ax.set_xticks(SELECTION_VALIDATION_NOISE_RATES)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.set_ylim(0, 1)
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.grid(axis="y", alpha=0.20)
        ax.set_axisbelow(True)
    fig.suptitle("Did Noisy-Validation Selection Improve, Preserve, or Reduce Test Accuracy?",
                 fontsize=14, y=0.97)
    fig.supxlabel("Validation labels flipped (%)", y=0.17)
    fig.supylabel("Proportion of simulations (%)", x=0.015, y=0.59)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.105),
               ncols=3, frameon=False, fontsize=10)
    fig.text(0.5, 0.045, "Each bar totals 30 simulations (100%). Numbers inside bars are repetition counts.\n"
             "Compare test accuracy with the same seed and training-noise level at 0% validation flips.\n"
             "The same accuracy can occur even when the selected depth changes.",
             ha="center", va="center", fontsize=10)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.89, bottom=0.25, hspace=0.28, wspace=0.15)
    fig.savefig(ADDITIONAL_SELECTION_FIGURE_PATHS[2], dpi=FIGURE_DPI)
    plt.close(fig)


def create_figure_12(raw):
    """Count paired wins, ties, and losses against unlimited depth; keep all seeds."""
    reference = raw.loc[raw["max_depth"] == "unlimited",
                        ["seed", "requested_noise_rate", "test_accuracy"]].rename(
                            columns={"test_accuracy": "unlimited_test_accuracy"})
    paired = raw.merge(reference, on=["seed", "requested_noise_rate"],
                       how="left", validate="many_to_one")
    require(len(paired) == EXPECTED_FITS and not paired["unlimited_test_accuracy"].isna().any(),
            "Figure 12 could not match every fit to its unlimited-depth reference.")
    differences = paired["test_accuracy"] - paired["unlimited_test_accuracy"]
    require(np.allclose(differences, paired[PAIRED_COLUMN], rtol=0, atol=NUMERIC_TOLERANCE),
            "Figure 12 paired differences disagree with the saved results.")
    paired["difference"] = differences
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 6.6), sharex=True, sharey=True)
    names = ("Lower test accuracy", "Equal test accuracy", "Higher test accuracy")
    for ax, label in zip(axes, DEPTH_LABELS[:-1]):
        counts = []
        for rate in NOISE_RATES:
            group = paired.loc[(paired["max_depth"] == label)
                               & (paired["requested_noise_rate"] == rate)]
            require(len(group) == len(REPETITION_SEEDS)
                    and set(group["seed"]) == set(REPETITION_SEEDS),
                    "Figure 12 requires all 20 paired repetitions in each condition.")
            values = group["difference"].to_numpy()
            # Tolerance handles CSV round-off, not a meaningful accuracy margin.
            # One prediction on the 300-point test set changes accuracy by 1/300.
            counts.append([int((values < -NUMERIC_TOLERANCE).sum()),
                           int((np.abs(values) <= NUMERIC_TOLERANCE).sum()),
                           int((values > NUMERIC_TOLERANCE).sum())])
        counts = np.asarray(counts)
        require((counts.sum(axis=1) == len(REPETITION_SEEDS)).all(),
                "Figure 12 win, tie, and loss counts must total 20 for each bar.")
        bottom = np.zeros(len(NOISE_RATES))
        for index, (name, color) in enumerate(zip(names, PERFORMANCE_OUTCOME_COLORS)):
            heights = counts[:, index] / len(REPETITION_SEEDS)
            ax.bar(NOISE_RATES, heights, bottom=bottom, width=0.065, color=color,
                   edgecolor="white", linewidth=0.7, label=name)
            for x, count, height, base in zip(NOISE_RATES, counts[:, index], heights, bottom):
                if count:
                    ax.text(x, base + height / 2, str(count), ha="center", va="center",
                            fontsize=10, color="#222222" if index == 1 else "white")
            bottom += heights
        ax.set_title(f"Depth {label} vs unlimited", color=DEPTH_COLORS[label], fontweight="bold")
        ax.set_xticks(NOISE_RATES)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.set_ylim(0, 1)
        ax.tick_params(labelleft=True)
        ax.grid(axis="y", alpha=0.20)
        ax.set_axisbelow(True)
    fig.suptitle("How Often Does a Depth-Limited Tree Beat an Unlimited Tree?", fontsize=16, y=0.97)
    fig.supxlabel("Training labels flipped (%)", y=0.23)
    fig.supylabel("Proportion of simulations (%)", x=0.015, y=0.61)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.14),
               ncols=3, frameon=False, fontsize=10)
    fig.text(0.5, 0.065, "Each bar counts all 20 simulations (100%); numbers inside bars are repetition counts.\n"
             "Compare test scores within the same seed and noise level. Test labels were never flipped.\n"
             "Winning more often does not measure the size of the advantage or establish statistical significance.",
             ha="center", va="center", fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.33, wspace=0.20)
    fig.savefig(ADDITIONAL_TRAIN_FIGURE_PATHS[3], dpi=FIGURE_DPI)
    plt.close(fig)


def create_figure_13(raw):
    """Compare original-label training accuracy with test accuracy for every fit."""
    # Both axes use unflipped reference labels. Noisy-label training accuracy is
    # different and remains available in Figure 2. A gap alone does not prove
    # overfitting; low training and test accuracy may also reflect underfitting.
    coordinates = raw[["train_accuracy_clean", "test_accuracy"]].to_numpy()
    require(np.isfinite(coordinates).all() and ((0 <= coordinates) & (coordinates <= 1)).all(),
            "Figure 13 needs finite accuracy values between zero and one.")
    # Common limits include EVERY repetition; equal x/y units make the diagonal meaningful.
    limits = (max(0.0, np.floor(coordinates.min() * 10) / 10 - 0.02), 1.02)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10), sharex=True, sharey=True)
    for ax, rate in zip(axes.flat, NOISE_RATES):
        ax.plot(limits, limits, color="#777777", linestyle="--", linewidth=1, zorder=1)
        for label in DEPTH_LABELS:
            group = raw.loc[(raw["requested_noise_rate"] == rate)
                            & (raw["max_depth"] == label)].sort_values("seed")
            require(len(group) == len(REPETITION_SEEDS)
                    and set(group["seed"]) == set(REPETITION_SEEDS),
                    "Figure 13 requires 20 simulations for every depth and noise level.")
            x, y = group["train_accuracy_clean"], group["test_accuracy"]
            # Exact plotted coordinates, without jitter: overlapping points stay overlapping.
            ax.scatter(x, y, color=DEPTH_COLORS[label], s=22, alpha=0.40,
                       edgecolors="none", zorder=2)
            ax.scatter([x.mean()], [y.mean()], color=DEPTH_COLORS[label], marker="D", s=80,
                       edgecolors="#222222", linewidths=0.9, zorder=3, label=f"Depth {label}")
        ax.set_title(f"Training labels flipped: {rate:.0%}")
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.grid(alpha=0.20)
        ax.set_axisbelow(True)
    fig.suptitle("Training and Test Performance against Original Labels", fontsize=16, y=0.97)
    fig.supxlabel("Training accuracy against original labels (%)", y=0.18)
    fig.supylabel("Accuracy on untouched test labels (%)", x=0.035, y=0.59)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.12),
               ncols=4, frameon=False, fontsize=10)
    fig.text(0.5, 0.055, "Small dots: 20 simulations per depth; dots may overlap. Diamonds: means. No uncertainty bars are shown.\n"
             "Dashed diagonal: equal training and test accuracy. A point below it has lower test accuracy.\n"
             "A gap alone does not prove overfitting. Read alongside Figure 2, which also scores noisy training labels.",
             ha="center", va="center", fontsize=10)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.89, bottom=0.27, hspace=0.30, wspace=0.13)
    fig.savefig(ADDITIONAL_TRAIN_FIGURE_PATHS[4], dpi=FIGURE_DPI)
    plt.close(fig)


def create_additional_figures_from_saved_results():
    """Read validated CSVs and create eight extra PNGs, with zero additional fits.

    main() calls this after the two experiments. To redraw only these eight figures,
    import this file and call this function; it never writes to the original CSVs.
    """
    print("\nCreating eight additional figures from existing CSVs; no additional fits.", flush=True)
    for path in (RAW_PATH, SUMMARY_PATH, SELECTION_RAW_PATH,
                 SELECTION_SUMMARY_PATH, SELECTION_FREQUENCIES_PATH):
        require(path.is_file() and path.stat().st_size > 0,
                f"cannot draw additional figures without the completed experiment output: {path}")
    raw = pd.read_csv(RAW_PATH, dtype={"max_depth": str})
    summary = pd.read_csv(SUMMARY_PATH, dtype={"max_depth": str})
    selection_raw = pd.read_csv(SELECTION_RAW_PATH, dtype={
        "selected_depth": str, "reference_depth_clean_validation": str})
    selection_summary = pd.read_csv(SELECTION_SUMMARY_PATH)
    frequencies = pd.read_csv(SELECTION_FREQUENCIES_PATH, dtype={"selected_depth": str})
    validate_results(raw, summary)
    validate_selection_results(selection_raw, selection_summary, frequencies)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SELECTION_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        create_figure_6(raw)
        create_figure_7(raw)
        create_figure_8(raw)
        create_figure_9(selection_summary)
        create_figure_10(selection_raw)
        create_figure_11(selection_raw)
        create_figure_12(raw)
        create_figure_13(raw)
    for path in (*ADDITIONAL_TRAIN_FIGURE_PATHS, *ADDITIONAL_SELECTION_FIGURE_PATHS):
        require(path.is_file() and path.stat().st_size > 0,
                f"additional figure is missing or empty: {path}")
        print(f"  {path}")
    print("Validation passed: all eight additional PNGs exist and are nonempty.", flush=True)


def main():
    started = perf_counter()
    print("Can Limiting Tree Depth Improve Robustness to Label Noise?", flush=True)
    print(f"Python {platform.python_version()} | NumPy {np.__version__} | pandas {pd.__version__} | "
          f"Matplotlib {matplotlib.__version__} | scikit-learn {sklearn.__version__}", flush=True)
    require(REPETITION_SEEDS == tuple(range(20)) and NOISE_RATES == (0.0, 0.1, 0.2, 0.3)
            and MAX_DEPTHS == (2, 4, 8, None), "the pre-specified experimental conditions changed.")
    require(N_SAMPLES == 1000 and FEATURE_NOISE == 0.20 and TEST_SIZE == 0.30,
            "the pre-specified data settings changed.")
    print("Fixed design: 20 simulations × 4 flip rates × 4 depths = 320 fits.\n"
          "Feature noise = 0.20; label flips affect training labels only.\n"
          "Clean means unflipped reference labels, not an unambiguous classification task.", flush=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    raw, illustration = run_repetitions()
    validate_results(raw, announce=False)
    summary = summarize_results(raw)
    validate_results(raw, summary)
    # Store proportions, not percentages. The default float formatting preserves
    # precision while keeping rates readable as 0.1, 0.2, and 0.3 in the CSVs.
    raw.to_csv(RAW_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    print("Creating three figures at 300 dpi; Figure 3 reuses fitted models.", flush=True)
    create_figure_1(summary)
    create_figure_2(summary)
    create_figure_3(illustration)
    validate_output_files()

    print("\nMean test accuracy (%) ± sample SD (percentage points):")
    display = summary.copy()
    display["result"] = display.apply(
        lambda row: f"{100 * row['test_accuracy_mean']:.2f} ± {100 * row['test_accuracy_std']:.2f}", axis=1
    )
    table = display.pivot(index="requested_noise_rate", columns="max_depth", values="result")
    table = table.reindex(index=NOISE_RATES, columns=DEPTH_LABELS)
    table.index = [f"{rate:.0%} flips" for rate in table.index]
    print(table.to_string())
    print("\nPaired test-accuracy difference vs unlimited at 30% flips (mean ± SD, percentage points):")
    for row in summary.loc[summary["requested_noise_rate"] == 0.30].to_dict("records"):
        print(f"  Depth {row['max_depth']:>9}: {100 * row[PAIRED_COLUMN + '_mean']:+.2f} "
              f"± {100 * row[PAIRED_COLUMN + '_std']:.2f}")
    print("SD describes variation across 20 simulations; it is not a confidence interval.\n"
          "Compare both absolute accuracy and its change from 0% flips: losing less accuracy\n"
          "does not guarantee higher final accuracy. These results compare fixed conditions;\n"
          "they do not establish a universal best depth or an unbiased test of a selected winner.")
    print("\nFinal output locations:")
    for path in (RAW_PATH, SUMMARY_PATH, *FIGURE_PATHS):
        print(f"  {path}")
    original_seconds = perf_counter() - started
    print(f"Original experiment elapsed time: {original_seconds:.2f} seconds.", flush=True)
    selection_started = perf_counter()
    run_validation_noise_experiment()
    print(f"Validation-noise experiment elapsed time: {perf_counter() - selection_started:.2f} seconds.")
    create_additional_figures_from_saved_results()
    print(f"Both experiments complete: {perf_counter() - started:.2f} seconds "
          "(fits, validation checks, CSV writing, and all thirteen figures; excludes imports).")


if __name__ == "__main__":
    main()
