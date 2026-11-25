"""Plot the dense baseline: Điều rank histogram and top-1 score spread.

Usage:
    uv run --with matplotlib python -m eval.plot_baseline \
        [--run eval/runs/2026-11-22-dense-baseline.jsonl] [--out docs/images]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"  # categorical slot 1
ORANGE = "#eb6834"  # categorical slot 2

RANK_BINS = ["1", "2", "3", "4", "5", "6–10", "trượt"]


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rank_bin(rank: int | None) -> str:
    if rank is None:
        return "trượt"
    if rank <= 5:
        return str(rank)
    return "6–10"


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, length=0)


def plot_rank_histogram(rows: list[dict], out: Path) -> None:
    groups = [("direct", "Câu trực tiếp", BLUE), ("paraphrase", "Câu đời thường", ORANGE)]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150, facecolor=SURFACE)
    style_axes(ax)
    width = 0.38
    for i, (group, label, color) in enumerate(groups):
        ranks = [rank_bin(r["rank_article"]) for r in rows if r["group"] == group]
        counts = [ranks.count(b) for b in RANK_BINS]
        xs = [x + (i - 0.5) * (width + 0.02) for x in range(len(RANK_BINS))]
        bars = ax.bar(xs, counts, width=width, color=color, label=f"{label} (n={len(ranks)})")
        for bar, n in zip(bars, counts, strict=True):
            if n:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    n + 0.6,
                    str(n),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=TEXT,
                )
    ax.set_xticks(range(len(RANK_BINS)), RANK_BINS)
    ax.set_xlabel("Hạng của chunk đầu tiên thuộc Điều đúng (top 10)", color=TEXT_2)
    ax.set_ylabel("Số câu hỏi", color=TEXT_2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Hạng mức Điều — dense baseline (bge-m3)", loc="left", color=TEXT, fontsize=12)
    ax.legend(frameon=False, labelcolor=TEXT, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_score_spread(rows: list[dict], out: Path) -> None:
    groups = [
        ("direct", "Trực tiếp"),
        ("paraphrase", "Đời thường"),
        ("article_ref", "Hỏi số Điều"),
        ("out_of_scope", "Ngoài phạm vi"),
    ]
    in_scope = [r["top"][0]["score"] for r in rows if r["group"] != "out_of_scope"]
    floor = min(in_scope)
    oos = [r["top"][0]["score"] for r in rows if r["group"] == "out_of_scope"]
    above = sum(s > floor for s in oos)

    rng = random.Random(0)
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150, facecolor=SURFACE)
    style_axes(ax)
    for y, (group, _label) in enumerate(groups):
        scores = [r["top"][0]["score"] for r in rows if r["group"] == group]
        color = ORANGE if group == "out_of_scope" else BLUE
        jitter = [y + rng.uniform(-0.18, 0.18) for _ in scores]
        ax.scatter(scores, jitter, s=34, color=color, edgecolors=SURFACE, linewidths=1.2, zorder=3)
    ax.axvline(floor, color=TEXT_2, linewidth=1, linestyle="--", zorder=2)
    ax.text(
        floor + 0.004,
        len(groups) - 0.45,
        f"điểm thấp nhất trong phạm vi = {floor:.3f}".replace(".", ","),
        color=TEXT_2,
        fontsize=9,
        va="center",
    )
    ax.set_yticks(range(len(groups)), [f"{lbl}" for _, lbl in groups])
    ax.set_ylim(-0.6, len(groups) - 0.2)
    ax.invert_yaxis()
    ax.set_xlabel("Điểm cosine của kết quả hạng 1", color=TEXT_2)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(
        f"Điểm top-1: {above}/{len(oos)} câu ngoài phạm vi vượt ngưỡng thấp nhất",
        loc="left",
        color=TEXT,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, default=Path("eval/runs/2026-11-22-dense-baseline.jsonl")
    )
    parser.add_argument("--out", type=Path, default=Path("docs/images"))
    args = parser.parse_args()
    rows = load_rows(args.run)
    args.out.mkdir(parents=True, exist_ok=True)
    plot_rank_histogram(rows, args.out / "baseline-rank-histogram.png")
    plot_score_spread(rows, args.out / "baseline-top1-scores.png")
    print(f"wrote 2 charts to {args.out}")


if __name__ == "__main__":
    main()
