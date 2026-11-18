"""Matplotlib figures for the panel: piano rolls on a shared time axis, and a chord strip."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {"before": "#c2410c", "after": "#0f766e", "ink": "#1f2937", "grid": "#e5e7eb", "speech": "#6d28d9"}


def piano_rolls(rows, duration_s, title=None):
    """rows: [(label, notes, colour_key, optional_track)] stacked vertically with common axes."""
    lo = min([n["pitch_midi"] for _, ns, *_ in rows for n in ns] or [48]) - 2
    hi = max([n["pitch_midi"] for _, ns, *_ in rows for n in ns] or [72]) + 2
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 1.9 * len(rows) + 0.4), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (label, notes, key, *rest) in zip(axes, rows):
        trk = rest[0] if rest else None
        if trk is not None:
            v = trk["voiced"]
            m = np.where(v, 69 + 12 * np.log2(np.where(v, trk["f0"], 440.0) / 440.0), np.nan)
            ax.plot(trk["t"], m, color="#9ca3af", lw=1.0, zorder=1)
        for n in notes:
            ax.barh(n["pitch_midi"], n["duration_s"], left=n["onset_s"], height=0.8, color=PALETTE[key], zorder=2)
        ax.set_ylim(lo, hi); ax.set_xlim(0, duration_s)
        ax.set_ylabel("MIDI"); ax.set_title(f"{label}  ({len(notes)} notes)", loc="left", fontsize=10, color=PALETTE["ink"])
        ax.grid(color=PALETTE["grid"], lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axes[-1].set_xlabel("seconds")
    if title: fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def chord_strip(chords, duration_s, max_s=None):
    fig, ax = plt.subplots(figsize=(10, 1.2))
    end = min(duration_s, max_s) if max_s else duration_s
    last = None
    for c in chords:
        if c["start_s"] >= end: break
        if c["label"] != last:
            ax.axvline(c["start_s"], color=PALETTE["grid"], lw=0.8)
            ax.text(c["start_s"] + 0.05, 0.5, c["label"], va="center", fontsize=9, color=PALETTE["ink"])
            last = c["label"]
    ax.set_xlim(0, end); ax.set_yticks([]); ax.set_xlabel("seconds")
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig
