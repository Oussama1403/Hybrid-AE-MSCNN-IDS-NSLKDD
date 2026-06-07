from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk")

NSL_KDD_COLUMNS: List[str] = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty",
]
FEATURE_COLUMNS: List[str] = NSL_KDD_COLUMNS[:41]
NUMERICAL_COLUMNS: List[str] = [c for c in FEATURE_COLUMNS if c not in {"protocol_type", "service", "flag"}]
CATEGORICAL_COLUMNS: List[str] = ["protocol_type", "service", "flag"]

ATTACK_FAMILY_MAP: Dict[str, str] = {
    "back": "DoS",
    "land": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "mailbomb": "DoS",
    "apache2": "DoS",
    "processtable": "DoS",
    "udpstorm": "DoS",
    "worm": "DoS",
    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "satan": "Probe",
    "mscan": "Probe",
    "saint": "Probe",
    "udpscan": "Probe",
    "guess_passwd": "R2L",
    "ftp_write": "R2L",
    "imap": "R2L",
    "phf": "R2L",
    "multihop": "R2L",
    "warezmaster": "R2L",
    "warezclient": "R2L",
    "spy": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    "snmpguess": "R2L",
    "snmpgetattack": "R2L",
    "httptunnel": "R2L",
    "sendmail": "R2L",
    "named": "R2L",
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "rootkit": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R",
    "ps": "U2R",
}

CLASS_ORDER = ["Normal", "DoS", "Probe", "R2L", "U2R"]


@dataclass
class DatasetSummary:
    train_rows: int
    test_rows: int
    train_normal: int
    train_anomaly: int
    test_normal: int
    test_anomaly: int
    train_normal_ratio: float
    train_anomaly_ratio: float
    test_normal_ratio: float
    test_anomaly_ratio: float
    test_unknown_attack_type_count: int
    test_zero_day_sample_count: int
    unknown_attack_types: List[str]


def resolve_dataset_dir(dataset_dir_arg: str) -> Path:
    raw = str(dataset_dir_arg).strip()
    if raw:
        provided = Path(raw).expanduser()
        if _has_dataset_pair(provided):
            return provided
        nested = provided / "nsl-kdd"
        if _has_dataset_pair(nested):
            return nested
        raise FileNotFoundError(
            f"Could not find KDDTrain+.txt and KDDTest+.txt under provided dataset-dir: {provided}"
        )

    candidates = [
        Path("NSL-KDD"),
        Path("NSL-KDD") / "nsl-kdd",
        Path("nsl-kdd"),
        Path("nslkdd"),
        Path("REPOS") / "NSL-KDD",
    ]
    for candidate in candidates:
        if _has_dataset_pair(candidate):
            return candidate

    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        for train_path in kaggle_root.rglob("KDDTrain+.txt"):
            parent = train_path.parent
            if _has_dataset_pair(parent):
                return parent

    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not auto-detect NSL-KDD dataset directory. Pass --dataset-dir explicitly.\n"
        f"Searched candidates:\n{searched}"
    )


def _has_dataset_pair(dataset_dir: Path) -> bool:
    return (dataset_dir / "KDDTrain+.txt").exists() and (dataset_dir / "KDDTest+.txt").exists()


def load_data(dataset_dir: Path | str):
    dataset_dir = Path(dataset_dir)
    train_path = dataset_dir / "KDDTrain+.txt"
    test_path = dataset_dir / "KDDTest+.txt"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected KDDTrain+.txt and KDDTest+.txt in {dataset_dir}"
        )

    train_df = pd.read_csv(train_path, names=NSL_KDD_COLUMNS, header=None)
    test_df = pd.read_csv(test_path, names=NSL_KDD_COLUMNS, header=None)
    train_df["label"] = train_df["label"].astype(str).str.strip().str.rstrip(".")
    test_df["label"] = test_df["label"].astype(str).str.strip().str.rstrip(".")
    return type(
        "DatasetBundle",
        (),
        {"train_df": train_df, "test_df": test_df, "train_path": train_path, "test_path": test_path},
    )()


def _ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _annotate_bars(ax: plt.Axes, fmt: str = "{:.0f}", color: str = "#1a1a1a") -> None:
    for bar in ax.patches:
        height = float(bar.get_height())
        if not np.isfinite(height):
            continue
        if height == 0:
            continue
        x = bar.get_x() + bar.get_width() / 2.0
        ax.annotate(
            fmt.format(height),
            (x, height),
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=11,
            color=color,
            fontweight="bold",
            clip_on=False,
        )


def _annotate_horizontal_bars(ax: plt.Axes, fmt: str = "{:.0f}", color: str = "#1a1a1a") -> None:
    for bar in ax.patches:
        width = float(bar.get_width())
        if not np.isfinite(width):
            continue
        if width == 0:
            continue
        y = bar.get_y() + bar.get_height() / 2.0
        ax.annotate(
            fmt.format(width),
            (width, y),
            ha="left",
            va="center",
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=11,
            color=color,
            fontweight="bold",
            clip_on=False,
        )


def _annotate_bars_with_percentages(
    ax: plt.Axes, plot_df: pd.DataFrame, group_col: str, value_col: str, fmt_val: str = "{:.0f}", color: str = "#1a1a1a"
) -> None:
    """Annotate bars with both absolute values and percentages within each group."""
    grouped = plot_df.groupby(group_col)[value_col].sum()
    categories = list(plot_df[group_col].drop_duplicates())

    for container in ax.containers:
        bars = list(container)
        if not bars:
            continue

        for bar, group_name in zip(bars, categories):
            height = float(bar.get_height())
            if not np.isfinite(height) or height == 0:
                continue

            x = bar.get_x() + bar.get_width() / 2.0
            y = height
            group_total = float(grouped.get(group_name, 0.0))
            pct = (height / group_total * 100.0) if group_total > 0 else 0.0
            label = f"{fmt_val.format(height)}\n({pct:.1f}%)"
            ax.annotate(
                label,
                (x, y),
                ha="center",
                va="bottom",
                xytext=(0, 4),
                textcoords="offset points",
                fontsize=10,
                color=color,
                fontweight="bold",
                clip_on=False,
            )


def _annotate_horizontal_bars_with_percentages(
    ax: plt.Axes, plot_df: pd.DataFrame, value_col: str, fmt_val: str = "{:.0f}", color: str = "#1a1a1a"
) -> None:
    """Annotate horizontal bars with both absolute values and percentages of total."""
    total = float(plot_df[value_col].sum())
    for bar in ax.patches:
        width = float(bar.get_width())
        if not np.isfinite(width) or width == 0:
            continue
        y = bar.get_y() + bar.get_height() / 2.0
        pct = (width / total * 100.0) if total > 0 else 0.0
        label = f"{fmt_val.format(width)} ({pct:.1f}%)"
        ax.annotate(
            label,
            (width, y),
            ha="left",
            va="center",
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=10,
            color=color,
            fontweight="bold",
            clip_on=False,
        )


def _annotate_grouped_bars_with_group_totals(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    x_col: str,
    hue_col: str,
    value_col: str,
    fmt_val: str = "{:.0f}",
    color: str = "#1a1a1a",
) -> None:
    """Annotate grouped vertical bars using explicit x/hue values and group totals."""
    x_order = list(pd.unique(plot_df[x_col]))
    hue_order = list(pd.unique(plot_df[hue_col]))
    totals = plot_df.groupby(x_col)[value_col].sum()

    for container_idx, container in enumerate(ax.containers):
        if container_idx >= len(hue_order):
            break
        hue_value = hue_order[container_idx]
        subset = plot_df[plot_df[hue_col] == hue_value].set_index(x_col)

        for bar, x_value in zip(container, x_order):
            height = float(bar.get_height())
            if not np.isfinite(height) or height == 0:
                continue

            x = bar.get_x() + bar.get_width() / 2.0
            y = height
            group_total = float(totals.get(x_value, 0.0))
            pct = (height / group_total * 100.0) if group_total > 0 else 0.0
            label = f"{fmt_val.format(height)}\n({pct:.1f}%)"
            ax.annotate(
                label,
                (x, y),
                ha="center",
                va="bottom",
                xytext=(0, 4),
                textcoords="offset points",
                fontsize=10,
                color=color,
                fontweight="bold",
                clip_on=False,
            )


def _table_block(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    return df.to_string(index=False)


def _family_label(label: str) -> str:
    label = str(label).strip().rstrip(".")
    if label == "normal":
        return "Normal"
    return ATTACK_FAMILY_MAP.get(label, "Unknown")


def _binary_split_summary(df: pd.DataFrame, split_name: str) -> Dict[str, float]:
    labels = df["label"].astype(str)
    normal = int(np.sum(labels == "normal"))
    anomaly = int(np.sum(labels != "normal"))
    total = int(len(df))
    return {
        "split": split_name,
        "samples": total,
        "normal": normal,
        "anomaly": anomaly,
        "normal_ratio": _safe_ratio(normal, total),
        "anomaly_ratio": _safe_ratio(anomaly, total),
        "normal_to_anomaly": _safe_ratio(normal, anomaly),
    }


def build_binary_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _binary_split_summary(train_df, "KDDTrain+"),
            _binary_split_summary(test_df, "KDDTest+"),
        ]
    )


def build_five_class_distribution(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    train_family = train_df["label"].astype(str).map(_family_label)
    test_family = test_df["label"].astype(str).map(_family_label)

    train_counts = train_family.value_counts().reindex(CLASS_ORDER, fill_value=0)
    test_counts = test_family.value_counts().reindex(CLASS_ORDER, fill_value=0)

    return pd.DataFrame(
        {
            "class": CLASS_ORDER,
            "train_count": [int(train_counts[c]) for c in CLASS_ORDER],
            "test_count": [int(test_counts[c]) for c in CLASS_ORDER],
        }
    )


def build_unknown_attack_report(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    train_labels = set(train_df["label"].astype(str).unique())
    test_labels = set(test_df["label"].astype(str).unique())
    unknown_attack_types = sorted((test_labels - train_labels) - {"normal"})

    rows = []
    for label in unknown_attack_types:
        count = int(np.sum(test_df["label"].astype(str) == label))
        rows.append({"attack_label": label, "count_in_test": count})
    return pd.DataFrame(rows).sort_values(["count_in_test", "attack_label"], ascending=[False, True])


def build_kddtest_label_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    train_labels = set(train_df["label"].astype(str).unique())
    test_labels = test_df["label"].astype(str)
    known_mask = test_labels.isin(train_labels)

    known_count = int(known_mask.sum())
    unknown_count = int((~known_mask).sum())
    total = int(len(test_df))

    return pd.DataFrame(
        [
            {
                "label_status": "Known (exists in KDDTrain+)",
                "sample_count": known_count,
                "sample_share": _safe_ratio(known_count, total),
                "sample_share_pct": _safe_ratio(known_count, total) * 100.0,
            },
            {
                "label_status": "Unknown (zero-day in KDDTest+)",
                "sample_count": unknown_count,
                "sample_share": _safe_ratio(unknown_count, total),
                "sample_share_pct": _safe_ratio(unknown_count, total) * 100.0,
            },
            {
                "label_status": "Total",
                "sample_count": total,
                "sample_share": 1.0,
                "sample_share_pct": 100.0,
            },
        ]
    )


def build_kddtest_label_table(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    train_labels = set(train_df["label"].astype(str).unique())
    counts = test_df["label"].astype(str).value_counts()

    rows = []
    for label, count in counts.items():
        is_unknown = label not in train_labels
        rows.append(
            {
                "label": label,
                "count_in_test": int(count),
                "status": "Unknown" if is_unknown else "Known",
                "is_unknown_in_train": int(is_unknown),
                "percent_of_test": _safe_ratio(int(count), int(len(test_df))) * 100.0,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["is_unknown_in_train", "count_in_test", "label"], ascending=[False, False, True]
    ).reset_index(drop=True)


def build_per_attack_label_counts(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    train_counts = train_df["label"].astype(str).value_counts()
    test_counts = test_df["label"].astype(str).value_counts()
    labels = sorted(set(train_counts.index) | set(test_counts.index))

    rows = []
    for label in labels:
        train_count = int(train_counts.get(label, 0))
        test_count = int(test_counts.get(label, 0))
        rows.append(
            {
                "attack_label": label,
                "train_count": train_count,
                "test_count": test_count,
                "is_unknown_in_train": int(train_count == 0 and label != "normal"),
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values(["is_unknown_in_train", "test_count", "attack_label"], ascending=[False, False, True]).reset_index(drop=True)


def build_numeric_shift_report(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    train_num = train_df[NUMERICAL_COLUMNS].apply(pd.to_numeric, errors="coerce")
    test_num = test_df[NUMERICAL_COLUMNS].apply(pd.to_numeric, errors="coerce")

    rows = []
    for feature in NUMERICAL_COLUMNS:
        tr = train_num[feature].astype(float)
        te = test_num[feature].astype(float)
        tr_mean = float(tr.mean())
        te_mean = float(te.mean())
        tr_var = float(tr.var(ddof=0))
        te_var = float(te.var(ddof=0))
        pooled_std = float(np.sqrt(max((tr_var + te_var) / 2.0, 1e-12)))
        rows.append(
            {
                "feature": feature,
                "train_mean": tr_mean,
                "test_mean": te_mean,
                "mean_diff": te_mean - tr_mean,
                "abs_mean_diff": abs(te_mean - tr_mean),
                "train_variance": tr_var,
                "test_variance": te_var,
                "variance_ratio_test_over_train": _safe_ratio(te_var, tr_var),
                "standardized_mean_shift": _safe_ratio(te_mean - tr_mean, pooled_std),
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values("abs_mean_diff", ascending=False).reset_index(drop=True)


def build_categorical_shift_report(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in ["protocol_type", "service", "flag"]:
        train_share = train_df[feature].astype(str).value_counts(normalize=True)
        test_share = test_df[feature].astype(str).value_counts(normalize=True)
        categories = sorted(set(train_share.index) | set(test_share.index))
        for category in categories:
            rows.append(
                {
                    "feature": feature,
                    "category": category,
                    "train_share": float(train_share.get(category, 0.0)),
                    "test_share": float(test_share.get(category, 0.0)),
                    "share_diff": float(test_share.get(category, 0.0) - train_share.get(category, 0.0)),
                }
            )
    df = pd.DataFrame(rows)
    df["abs_share_diff"] = df["share_diff"].abs()
    return df.sort_values(["feature", "abs_share_diff"], ascending=[True, False]).reset_index(drop=True)


def build_feature_family_report(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in ["protocol_type", "service", "flag"]:
        for split_name, df in [("KDDTrain+", train_df), ("KDDTest+", test_df)]:
            counts = df[feature].astype(str).value_counts()
            top = counts.head(10)
            for value, count in top.items():
                rows.append(
                    {
                        "split": split_name,
                        "feature": feature,
                        "value": value,
                        "count": int(count),
                        "share": float(count / len(df)),
                    }
                )
    return pd.DataFrame(rows)


def save_tables(output_dir: Path, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    binary_summary = build_binary_summary(train_df, test_df)
    five_class = build_five_class_distribution(train_df, test_df)
    unknown_attack = build_unknown_attack_report(train_df, test_df)
    kddtest_label_summary = build_kddtest_label_summary(train_df, test_df)
    kddtest_label_table = build_kddtest_label_table(train_df, test_df)
    per_attack_counts = build_per_attack_label_counts(train_df, test_df)
    numeric_shift = build_numeric_shift_report(train_df, test_df)
    categorical_shift = build_categorical_shift_report(train_df, test_df)
    feature_family = build_feature_family_report(train_df, test_df)

    _save_csv(binary_summary, output_dir / "binary_split_summary.csv")
    _save_csv(five_class, output_dir / "five_class_distribution.csv")
    _save_csv(unknown_attack, output_dir / "unknown_attack_types.csv")
    _save_csv(kddtest_label_summary, output_dir / "kddtest_label_known_unknown_summary.csv")
    _save_csv(kddtest_label_table, output_dir / "kddtest_label_table_highlight_unknowns.csv")
    _save_csv(per_attack_counts, output_dir / "per_attack_label_counts_train_vs_test.csv")
    _save_csv(numeric_shift, output_dir / "numeric_feature_shift.csv")
    _save_csv(categorical_shift, output_dir / "categorical_feature_shift.csv")
    _save_csv(feature_family, output_dir / "top_categorical_values.csv")

    return {
        "binary_summary": binary_summary,
        "five_class": five_class,
        "unknown_attack": unknown_attack,
        "kddtest_label_summary": kddtest_label_summary,
        "kddtest_label_table": kddtest_label_table,
        "per_attack_counts": per_attack_counts,
        "numeric_shift": numeric_shift,
        "categorical_shift": categorical_shift,
        "feature_family": feature_family,
    }


def plot_binary_split_summary(output_dir: Path, binary_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = binary_summary.melt(id_vars="split", value_vars=["normal", "anomaly"], var_name="traffic", value_name="count")
    plot_df["traffic"] = plot_df["traffic"].replace({"normal": "Normal", "anomaly": "Anomaly"})
    sns.barplot(data=plot_df, x="split", y="count", hue="traffic", ax=ax, palette=["#2E86AB", "#E45756"])
    ax.set_title("Normal vs Anomaly Traffic: KDDTrain+ vs KDDTest+")
    ax.set_xlabel("")
    ax.set_ylabel("Samples")
    ax.legend(title="Traffic")

    _annotate_grouped_bars_with_group_totals(ax, plot_df, x_col="split", hue_col="traffic", value_col="count", fmt_val="{:.0f}")
    _save_figure(fig, output_dir / "01_binary_traffic_counts.png")


def plot_binary_ratios(output_dir: Path, binary_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df = binary_summary.melt(id_vars="split", value_vars=["normal_ratio", "anomaly_ratio"], var_name="traffic", value_name="ratio")
    plot_df["traffic"] = plot_df["traffic"].replace({"normal_ratio": "Normal", "anomaly_ratio": "Anomaly"})
    sns.barplot(data=plot_df, x="split", y="ratio", hue="traffic", ax=ax, palette=["#2E86AB", "#E45756"])
    ax.set_title("Raw Traffic Ratios in Each Split")
    ax.set_xlabel("")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1)
    ax.legend(title="Traffic")
    _annotate_bars(ax, fmt="{:.2f}")
    _save_figure(fig, output_dir / "02_binary_traffic_ratios.png")


def plot_five_class_distribution(output_dir: Path, five_class: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df = five_class.melt(id_vars="class", value_vars=["train_count", "test_count"], var_name="split", value_name="count")
    plot_df["split"] = plot_df["split"].replace({"train_count": "KDDTrain+", "test_count": "KDDTest+"})
    sns.barplot(data=plot_df, x="class", y="count", hue="split", ax=ax, palette=["#2E86AB", "#E45756"])
    ax.set_title("5-Class Distribution: Normal, DoS, Probe, R2L, U2R")
    ax.set_xlabel("")
    ax.set_ylabel("Samples")
    ax.legend(title="Split")

    _annotate_grouped_bars_with_group_totals(ax, plot_df, x_col="class", hue_col="split", value_col="count", fmt_val="{:.0f}")
    _save_figure(fig, output_dir / "03_five_class_distribution.png")


def plot_unknown_attacks(output_dir: Path, unknown_attack: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    if unknown_attack.empty:
        ax.text(0.5, 0.5, "No unknown attacks found", ha="center", va="center", fontsize=16)
        ax.axis("off")
    else:
        sns.barplot(data=unknown_attack, y="attack_label", x="count_in_test", ax=ax, color="#59A14F")
        ax.set_title("Unknown Attack Labels in KDDTest+ but Missing from KDDTrain+")
        ax.set_xlabel("Test Samples")
        ax.set_ylabel("Attack Label")
        _annotate_horizontal_bars(ax, fmt="{:.0f}")
    _save_figure(fig, output_dir / "04_unknown_attack_labels.png")


def plot_numeric_mean_shift(output_dir: Path, numeric_shift: pd.DataFrame, top_n: int = 20) -> None:
    top = numeric_shift.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(12, 8))
    top["direction"] = np.where(top["standardized_mean_shift"] >= 0, "positive", "negative")
    sns.barplot(data=top, y="feature", x="standardized_mean_shift", hue="direction", ax=ax, palette={"positive": "#E45756", "negative": "#2E86AB"}, dodge=False)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(f"Top {top_n} Numerical Features by Standardized Train-Test Mean Shift")
    ax.set_xlabel("Standardized Mean Shift (Test - Train)")
    ax.set_ylabel("")
    if ax.legend_ is not None:
        ax.legend_.remove()
    _annotate_horizontal_bars(ax, fmt="{:.2f}")
    _save_figure(fig, output_dir / "05_numeric_mean_shift_top20.png")


def plot_numeric_variance_ratio(output_dir: Path, numeric_shift: pd.DataFrame, top_n: int = 20) -> None:
    top = numeric_shift.assign(abs_ratio_dist=(numeric_shift["variance_ratio_test_over_train"] - 1.0).abs()).sort_values(
        "abs_ratio_dist", ascending=False
    ).head(top_n)
    fig, ax = plt.subplots(figsize=(12, 8))
    top["variance_side"] = np.where(top["variance_ratio_test_over_train"] >= 1.0, "test_higher", "train_higher")
    sns.barplot(
        data=top,
        y="feature",
        x="variance_ratio_test_over_train",
        hue="variance_side",
        ax=ax,
        palette={"test_higher": "#59A14F", "train_higher": "#F28E2B"},
        dodge=False,
    )
    ax.axvline(1.0, color="black", linewidth=1)
    ax.set_title(f"Top {top_n} Numerical Features by Variance Ratio (Test / Train)")
    ax.set_xlabel("Variance Ratio")
    ax.set_ylabel("")
    if ax.legend_ is not None:
        ax.legend_.remove()
    _annotate_horizontal_bars(ax, fmt="{:.2f}")
    _save_figure(fig, output_dir / "06_numeric_variance_ratio_top20.png")


def plot_per_attack_label_counts(output_dir: Path, per_attack_counts: pd.DataFrame, top_n: int = 30) -> None:
    top = per_attack_counts.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(14, 10))

    y_pos = np.arange(len(top))
    unknown_mask = top["is_unknown_in_train"].to_numpy().astype(bool)

    train_color = np.where(unknown_mask, "#bdbdbd", "#4C78A8")
    test_color = np.where(unknown_mask, "#D62728", "#E45756")

    bar_h = 0.38
    train_bars = ax.barh(y_pos - bar_h / 2, top["train_count"], height=bar_h, color=train_color, label="Train")
    test_bars = ax.barh(y_pos + bar_h / 2, top["test_count"], height=bar_h, color=test_color, label="Test")

    # Calculate totals for percentage
    train_total = float(top["train_count"].sum())
    test_total = float(top["test_count"].sum())
    
    for i, (_, row) in enumerate(top.iterrows()):
        train_cnt = float(row["train_count"])
        test_cnt = float(row["test_count"])
        train_pct = (train_cnt / train_total * 100.0) if train_total > 0 else 0.0
        test_pct = (test_cnt / test_total * 100.0) if test_total > 0 else 0.0
        
        train_label = f"{int(train_cnt)} ({train_pct:.1f}%)"
        test_label = f"{int(test_cnt)} ({test_pct:.1f}%)"
        
        ax.text(train_cnt + 5, i - bar_h / 2, train_label, va="center", ha="left", fontsize=8, color="#1a1a1a", fontweight="bold")
        ax.text(test_cnt + 5, i + bar_h / 2, test_label, va="center", ha="left", fontsize=8, color="#1a1a1a", fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["attack_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_ylabel("Attack Label")
    ax.set_title(f"Per-Attack Label Counts in Train vs Test (Top {top_n}; Unknown-in-Train Highlighted in Red)")
    ax.legend(title="Split")
    _save_figure(fig, output_dir / "11_per_attack_label_counts_train_vs_test.png")


def plot_five_class_donut(output_dir: Path, five_class: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]

    train_values = five_class["train_count"].to_numpy()
    test_values = five_class["test_count"].to_numpy()
    labels = five_class["class"].tolist()

    def _fmt_pct(values: np.ndarray):
        total = float(np.sum(values))
        return lambda pct: f"{pct:.1f}%\n({int(round(pct * total / 100.0)):,})"

    wedges_train, *_ = axes[0].pie(
        train_values,
        labels=labels,
        autopct=_fmt_pct(train_values),
        startangle=90,
        colors=colors,
        wedgeprops={"width": 0.42, "edgecolor": "white"},
        textprops={"fontsize": 10},
    )
    axes[0].set_title("KDDTrain+ 5-Class Doughnut")
    axes[0].axis("equal")

    wedges_test, *_ = axes[1].pie(
        test_values,
        labels=labels,
        autopct=_fmt_pct(test_values),
        startangle=90,
        colors=colors,
        wedgeprops={"width": 0.42, "edgecolor": "white"},
        textprops={"fontsize": 10},
    )
    axes[1].set_title("KDDTest+ 5-Class Doughnut")
    axes[1].axis("equal")

    fig.legend(wedges_train, labels, loc="lower center", ncol=5, frameon=False)
    _save_figure(fig, output_dir / "12_five_class_doughnut_train_test.png")


def plot_numerical_correlation_heatmaps(output_dir: Path, train_df: pd.DataFrame, test_df: pd.DataFrame, top_n: int = 20) -> None:
    numeric_shift = build_numeric_shift_report(train_df, test_df)
    features = numeric_shift.head(top_n)["feature"].tolist()

    train_num = train_df[features].apply(pd.to_numeric, errors="coerce")
    test_num = test_df[features].apply(pd.to_numeric, errors="coerce")
    corr_train = train_num.corr(numeric_only=True)
    corr_test = test_num.corr(numeric_only=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    sns.heatmap(corr_train, cmap="coolwarm", center=0.0, vmin=-1, vmax=1, ax=axes[0], cbar=False)
    axes[0].set_title("Train Correlation Heatmap (Top Shifted Numerical Features)")
    axes[0].tick_params(axis="x", labelrotation=80, labelsize=8)
    axes[0].tick_params(axis="y", labelsize=8)

    sns.heatmap(corr_test, cmap="coolwarm", center=0.0, vmin=-1, vmax=1, ax=axes[1], cbar=True)
    axes[1].set_title("Test Correlation Heatmap (Top Shifted Numerical Features)")
    axes[1].tick_params(axis="x", labelrotation=80, labelsize=8)
    axes[1].tick_params(axis="y", labelsize=8)

    _save_figure(fig, output_dir / "13_numerical_correlation_heatmaps_train_vs_test.png")


def save_zero_day_glance_table(output_dir: Path, unknown_attack: pd.DataFrame) -> None:
    compact = unknown_attack.copy()
    if compact.empty:
        compact = pd.DataFrame([{"attack_label": "(none)", "count_in_test": 0, "percent_of_total": 0.0}])
    else:
        # Add percentage column
        total = float(compact["count_in_test"].sum())
        compact["percent_of_total"] = (compact["count_in_test"] / total * 100.0) if total > 0 else 0.0
    
    compact_export = compact[["attack_label", "count_in_test", "percent_of_total"]].copy()
    _save_csv(compact_export, output_dir / "14_zero_day_glance_table.csv")

    fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(compact) + 2)))
    ax.axis("off")
    ax.set_title("Zero-Day at a Glance (KDDTest+ labels absent from KDDTrain+)", fontsize=12, pad=12)

    # Prepare table data with percentages
    table_data = []
    for _, row in compact.iterrows():
        if "percent_of_total" in row:
            table_data.append([row["attack_label"], f"{int(row['count_in_test'])} ({row['percent_of_total']:.1f}%)"])
        else:
            table_data.append([row["attack_label"], str(int(row["count_in_test"]))])
    
    col_labels = ["Unknown Attack Label", "Count in Test (%)"]
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.25)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2E86AB")
        else:
            cell.set_facecolor("#F8F9FA" if r % 2 else "#E9ECEF")

    _save_figure(fig, output_dir / "14_zero_day_glance_table.png")


def plot_class_imbalance_waterfall(output_dir: Path, binary_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(binary_summary))
    normal = binary_summary["normal"].to_numpy()
    anomaly = binary_summary["anomaly"].to_numpy()
    total = binary_summary["samples"].to_numpy()

    ax.bar(x, normal, color="#4C78A8", label="Normal")
    ax.bar(x, anomaly, bottom=normal, color="#E45756", label="Anomaly")

    for i in range(len(x)):
        total_val = float(total[i])
        normal_val = float(normal[i])
        anomaly_val = float(anomaly[i])
        
        normal_pct = (normal_val / total_val * 100.0) if total_val > 0 else 0.0
        anomaly_pct = (anomaly_val / total_val * 100.0) if total_val > 0 else 0.0
        
        # Total label at the top
        ax.text(x[i], total_val + 500, f"Total: {int(total_val):,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        
        # Normal percentage in the middle of normal segment
        ax.text(x[i], normal_val / 2.0, f"Normal\n{int(normal_val):,}\n({normal_pct:.1f}%)", 
                ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        
        # Anomaly percentage in the middle of anomaly segment
        ax.text(x[i], normal_val + anomaly_val / 2.0, f"Anomaly\n{int(anomaly_val):,}\n({anomaly_pct:.1f}%)", 
                ha="center", va="center", fontsize=9, fontweight="bold", color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(binary_summary["split"])
    ax.set_ylabel("Samples")
    ax.set_title("Class Imbalance Snapshot (Stacked Totals by Split)")
    ax.legend(title="Traffic")
    _save_figure(fig, output_dir / "15_class_imbalance_stacked_totals.png")


def plot_feature_drift_heatmap(output_dir: Path, train_df: pd.DataFrame, test_df: pd.DataFrame, top_features: List[str]) -> None:
    train_num = train_df[top_features].apply(pd.to_numeric, errors="coerce")
    test_num = test_df[top_features].apply(pd.to_numeric, errors="coerce")
    summary = pd.DataFrame(
        {
            "Train Mean": train_num.mean(axis=0),
            "Test Mean": test_num.mean(axis=0),
            "Train Var": train_num.var(axis=0, ddof=0),
            "Test Var": test_num.var(axis=0, ddof=0),
        }
    )
    summary["Mean Shift"] = summary["Test Mean"] - summary["Train Mean"]
    summary["Variance Ratio"] = summary["Test Var"] / summary["Train Var"].replace(0, np.nan)
    summary = summary.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(summary[["Train Mean", "Test Mean", "Mean Shift", "Variance Ratio"]], cmap="coolwarm", ax=ax)
    ax.set_title("Feature Drift Snapshot: Means and Variance Ratios")
    ax.set_xlabel("")
    ax.set_ylabel("")
    _save_figure(fig, output_dir / "07_feature_drift_heatmap.png")


def plot_categorical_share_shift(output_dir: Path, categorical_shift: pd.DataFrame) -> None:
    for feature in ["protocol_type", "service", "flag"]:
        subset = categorical_shift[categorical_shift["feature"] == feature].copy()
        if subset.empty:
            continue
        subset = subset.head(12)
        fig, ax = plt.subplots(figsize=(11, 6))
        plot_df = subset.melt(id_vars="category", value_vars=["train_share", "test_share"], var_name="split", value_name="share")
        plot_df["split"] = plot_df["split"].replace({"train_share": "Train", "test_share": "Test"})
        sns.barplot(data=plot_df, y="category", x="share", hue="split", ax=ax, palette=["#2E86AB", "#E45756"])
        ax.set_title(f"Top Categorical Distribution Shifts: {feature}")
        ax.set_xlabel("Share of Split")
        ax.set_ylabel("")
        _annotate_horizontal_bars(ax, fmt="{:.2f}")
        _save_figure(fig, output_dir / f"08_{feature}_share_shift.png")


def plot_top_feature_boxplots(output_dir: Path, train_df: pd.DataFrame, test_df: pd.DataFrame, numeric_shift: pd.DataFrame, top_n: int = 6) -> None:
    top_features = numeric_shift.head(top_n)["feature"].tolist()
    for feature in top_features:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_df = pd.DataFrame(
            {
                "value": pd.concat([train_df[feature].astype(float), test_df[feature].astype(float)], axis=0, ignore_index=True),
                "split": ["Train"] * len(train_df) + ["Test"] * len(test_df),
            }
        )
        sns.boxplot(data=plot_df, x="split", y="value", ax=ax, palette=["#2E86AB", "#E45756"])
        ax.set_title(f"Train vs Test Distribution: {feature}")
        ax.set_xlabel("")
        ax.set_ylabel(feature)
        _save_figure(fig, output_dir / f"09_boxplot_{feature}.png")

def plot_top_feature_boxplots_combined(
    output_dir: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_shift: pd.DataFrame,
    top_n: int = 5,
) -> None:
    top_features = numeric_shift.head(top_n)["feature"].tolist()
    fig, axes = plt.subplots(1, top_n, figsize=(4 * top_n, 5))
    if top_n == 1:
        axes = [axes]  # ensure iterable

    palette = {"Train": "#2E86AB", "Test": "#E45756"}
    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        plot_df = pd.DataFrame(
            {
                "value": pd.concat(
                    [train_df[feature].astype(float), test_df[feature].astype(float)],
                    axis=0,
                    ignore_index=True,
                ),
                "Split": ["Train"] * len(train_df) + ["Test"] * len(test_df),
            }
        )
        sns.boxplot(data=plot_df, x="Split", y="value", hue="Split",
                    ax=ax, palette=palette, legend=False)
        ax.set_title(feature, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("" if idx > 0 else "Value", fontsize=9)
        ax.tick_params(axis="x", labelsize=9)

    fig.suptitle(f"Top {top_n} Numerical Features with Largest Train‑Test Distribution Shift",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    _save_figure(fig, output_dir / "16_top_feature_shift_combined.png")


def plot_feature_family_snapshot(output_dir: Path, feature_family: pd.DataFrame) -> None:
    for feature in ["protocol_type", "service", "flag"]:
        subset = feature_family[feature_family["feature"] == feature].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(["split", "count"], ascending=[True, False]).groupby("split").head(8)
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_df = subset.copy()
        sns.barplot(data=plot_df, y="value", x="count", hue="split", ax=ax)
        ax.set_title(f"Most Frequent {feature} Values in Train vs Test")
        ax.set_xlabel("Count")
        ax.set_ylabel("")
        _annotate_horizontal_bars(ax, fmt="{:.0f}")
        _save_figure(fig, output_dir / f"10_top_values_{feature}.png")


def write_report_markdown(output_dir: Path, summary: DatasetSummary, tables: Dict[str, pd.DataFrame]) -> None:
    binary_summary = tables["binary_summary"]
    five_class = tables["five_class"]
    unknown_attack = tables["unknown_attack"]
    kddtest_label_summary = tables["kddtest_label_summary"]
    kddtest_label_table = tables["kddtest_label_table"]
    per_attack_counts = tables["per_attack_counts"]
    numeric_shift = tables["numeric_shift"]

    top_numeric = numeric_shift.head(12)[["feature", "standardized_mean_shift", "variance_ratio_test_over_train"]]

    lines = [
        "# NSL-KDD Dataset Research Report",
        "",
        "## Executive Summary",
        f"- KDDTrain+ samples: {summary.train_rows:,}",
        f"- KDDTest+ samples: {summary.test_rows:,}",
        f"- Zero-day / unknown attack types in test: {summary.test_unknown_attack_type_count}",
        f"- Zero-day attack samples in test: {summary.test_zero_day_sample_count:,}",
        "",
        "## Binary Traffic Split",
        _table_block(binary_summary),
        "",
        "## 5-Class Distribution",
        _table_block(five_class),
        "",
        "## Unknown Attack Labels",
        _table_block(unknown_attack),
        "",
        "## KDDTest+ Known vs Unknown Label Summary",
        _table_block(kddtest_label_summary),
        "",
        "## KDDTest+ Label Table",
        _table_block(kddtest_label_table),
        "",
        "## Per-Attack Label Train vs Test (Top Rows)",
        _table_block(per_attack_counts.head(20)),
        "",
        "## Top Numerical Feature Shifts",
        _table_block(top_numeric),
        "",
        "## Suggested Presentation Angle",
        "- Emphasize the strict train/test protocol and the presence of unseen attacks in KDDTest+.",
        "- Use the 5-class table to show how the benchmark compresses many labels into Normal, DoS, Probe, R2L, and U2R.",
        "- Use the numerical drift plots to argue that the test distribution is not trivial and is a meaningful generalization challenge.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_summary_json(output_dir: Path, summary: DatasetSummary) -> None:
    (output_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")


def build_report(dataset_dir: Path, output_dir: Path) -> None:
    _ensure_output_dir(output_dir)
    data = load_data(dataset_dir)
    train_df = data.train_df.copy()
    test_df = data.test_df.copy()

    tables = save_tables(output_dir, train_df, test_df)
    unknown_attack = tables["unknown_attack"]
    per_attack_counts = tables["per_attack_counts"]
    binary_summary = tables["binary_summary"]
    five_class = tables["five_class"]
    numeric_shift = tables["numeric_shift"]
    categorical_shift = tables["categorical_shift"]
    feature_family = tables["feature_family"]

    summary = DatasetSummary(
        train_rows=int(len(train_df)),
        test_rows=int(len(test_df)),
        train_normal=int(binary_summary.loc[binary_summary["split"] == "KDDTrain+", "normal"].iloc[0]),
        train_anomaly=int(binary_summary.loc[binary_summary["split"] == "KDDTrain+", "anomaly"].iloc[0]),
        test_normal=int(binary_summary.loc[binary_summary["split"] == "KDDTest+", "normal"].iloc[0]),
        test_anomaly=int(binary_summary.loc[binary_summary["split"] == "KDDTest+", "anomaly"].iloc[0]),
        train_normal_ratio=float(binary_summary.loc[binary_summary["split"] == "KDDTrain+", "normal_ratio"].iloc[0]),
        train_anomaly_ratio=float(binary_summary.loc[binary_summary["split"] == "KDDTrain+", "anomaly_ratio"].iloc[0]),
        test_normal_ratio=float(binary_summary.loc[binary_summary["split"] == "KDDTest+", "normal_ratio"].iloc[0]),
        test_anomaly_ratio=float(binary_summary.loc[binary_summary["split"] == "KDDTest+", "anomaly_ratio"].iloc[0]),
        test_unknown_attack_type_count=int(len(unknown_attack)),
        test_zero_day_sample_count=int(unknown_attack["count_in_test"].sum()) if not unknown_attack.empty else 0,
        unknown_attack_types=unknown_attack["attack_label"].tolist(),
    )

    plot_binary_split_summary(output_dir, binary_summary)
    plot_binary_ratios(output_dir, binary_summary)
    plot_five_class_distribution(output_dir, five_class)
    plot_unknown_attacks(output_dir, unknown_attack)
    plot_numeric_mean_shift(output_dir, numeric_shift)
    plot_numeric_variance_ratio(output_dir, numeric_shift)
    plot_feature_drift_heatmap(output_dir, train_df, test_df, numeric_shift.head(12)["feature"].tolist())
    plot_categorical_share_shift(output_dir, categorical_shift)
    plot_top_feature_boxplots(output_dir, train_df, test_df, numeric_shift, top_n=6)
    plot_feature_family_snapshot(output_dir, feature_family)
    plot_per_attack_label_counts(output_dir, per_attack_counts, top_n=35)
    plot_five_class_donut(output_dir, five_class)
    plot_numerical_correlation_heatmaps(output_dir, train_df, test_df, top_n=20)
    save_zero_day_glance_table(output_dir, unknown_attack)
    plot_class_imbalance_waterfall(output_dir, binary_summary)
    plot_top_feature_boxplots_combined(output_dir, train_df, test_df, numeric_shift, top_n=5)

    write_report_markdown(output_dir, summary, tables)
    write_summary_json(output_dir, summary)

    print(f"[NSL-KDD Report] Saved outputs to: {output_dir}")
    print(f"[NSL-KDD Report] Unknown attack labels: {', '.join(summary.unknown_attack_types)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate research-grade NSL-KDD dataset plots and tables for a presentation slide."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="",
        help="Folder containing KDDTrain+.txt and KDDTest+.txt. If omitted, the script auto-detects common locations.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="nslkdd_dataset_report",
        help="Directory where plots, tables, and summary files are written.",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    dataset_dir = resolve_dataset_dir(args.dataset_dir)
    output_dir = Path(args.output_dir).expanduser()
    build_report(dataset_dir=dataset_dir, output_dir=output_dir)


if __name__ == "__main__":
    main()
