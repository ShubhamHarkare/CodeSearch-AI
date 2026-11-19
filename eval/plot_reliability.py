"""Draws the README's reliability diagram from the calibration report.

    uv run python -m eval.fit_calibrator results/answers_dense.json  # -> results/calibration.json
    uv run python -m eval.plot_reliability                           # -> docs/reliability.png

One point per confidence bin: x = mean predicted confidence, y = share of answers that
were actually good, labelled with how many answers the bin holds. A perfectly
calibrated model sits on the diagonal. Bin sizes are shown because with ~50 answers a
single answer moves a small bin a lot.
"""

import json
from pathlib import Path
from typing import Annotated, Any

import typer

# Reference palette (dataviz skill): one series, so slot 1 only; recessive ink for the rest.
SURFACE = "#fcfcfb"
SERIES = "#2a78d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def plot(report: dict[str, Any], out: Path) -> Path:
    """Render the reliability diagram for a fit_calibrator report to a PNG."""
    import matplotlib

    matplotlib.use("Agg")  # no window: render straight to a file
    import matplotlib.pyplot as plt

    bins = report["bins"]
    xs = [b["confidence"] for b in bins]
    ys = [b["accuracy"] for b in bins]

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Perfect calibration: the reference, recessive and labelled directly.
    ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(  # follows the diagonal at any aspect ratio
        0.22,
        0.25,
        "perfect calibration",
        color=TEXT_SECONDARY,
        fontsize=8,
        rotation=45,
        transform_rotates_text=True,
        rotation_mode="anchor",
        ha="center",
        va="center",
    )

    # The model: a 2px line through >= 8px markers with a surface ring.
    ax.plot(xs, ys, color=SERIES, linewidth=2, zorder=2)
    ax.scatter(xs, ys, s=70, color=SERIES, edgecolors=SURFACE, linewidths=2, zorder=3)
    for x, y, b in zip(xs, ys, bins, strict=True):
        ax.annotate(
            f"n={b['count']}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
            color=TEXT_PRIMARY,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlabel("Predicted confidence", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("Share of answers that were good", color=TEXT_SECONDARY, fontsize=9)
    ax.set_title(
        f"Confidence calibration — {report['n']} answers, out-of-fold, ECE {report['ece']:.2f}",
        color=TEXT_PRIMARY,
        fontsize=10,
        loc="left",
    )
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def main(
    report: Annotated[Path, typer.Argument(exists=True, dir_okay=False)] = Path(
        "results/calibration.json"
    ),
    out: Annotated[Path, typer.Option(help="Where to write the PNG.")] = Path(
        "docs/reliability.png"
    ),
) -> None:
    """Draw the reliability diagram."""
    typer.echo(f"wrote {plot(json.loads(report.read_text()), out)}")


if __name__ == "__main__":
    typer.run(main)
