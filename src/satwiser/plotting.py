"""Shared Matplotlib styling for the report figures (matches the app's design tokens)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BG = "#06080E"
HAIRLINE = "#161B2A"
BORDER = "#1E2436"
TEXT = "#E8ECF4"
MUTED = "#8C95AB"
FAINT = "#5E6782"
DATA = "#9CC2FF"
EVENT = "#F4A259"


def style(ax) -> None:
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BORDER)
    ax.tick_params(colors=FAINT, labelsize=8)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    ax.grid(color=HAIRLINE, lw=0.6)


def figure(rows: int = 1, cols: int = 1, width: float = 11, height: float = 3.0, **kwargs):
    fig, axes = plt.subplots(rows, cols, figsize=(width, height * rows), squeeze=False,
                             **kwargs)
    fig.patch.set_facecolor(BG)
    for ax in axes.ravel():
        style(ax)
    return fig, axes


def legend(ax, **kwargs) -> None:
    leg = ax.legend(frameon=False, fontsize=8, **kwargs)
    for text in leg.get_texts():
        text.set_color(MUTED)


def save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=BG)
    plt.close(fig)
