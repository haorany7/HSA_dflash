"""Plot throughput vs per-user TPS from pareto.py results: one curve per system.

Usage: .venv/bin/python scripts/hetero/plot_pareto.py [docs/data/pareto.jsonl] [docs/pareto.png]
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else "docs/data/pareto.jsonl"
dst = sys.argv[2] if len(sys.argv) > 2 else "docs/pareto.png"
rows = [json.loads(line) for line in open(src)]

# Color = system (fixed categorical order).
SYSTEMS = [
    ("ar", "AR (GPU W8A8)", "#2a78d6"),
    ("gpu", "DFlash, all GPU", "#eb6834"),
    ("hetero", "DFlash, GPU target + NPU draft", "#1baf7a"),
]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def best_per_batch(points):
    """One point per concurrency: the block size with the highest throughput at that batch.

    At a fixed batch, throughput = batch * per-user TPS, so this block is also the
    fastest per user; the frontier of these points is the system's tuned curve.
    """
    best = {}
    for r in points:
        if r["batch"] not in best or r["throughput"] > best[r["batch"]]["throughput"]:
            best[r["batch"]] = r
    return [best[b] for b in sorted(best)]


fig, ax = plt.subplots(figsize=(8, 5.2), dpi=160)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
for key, label, color in SYSTEMS:
    pts = best_per_batch([r for r in rows if r["system"] == key])
    ax.plot([r["tps_per_user"] for r in pts], [r["throughput"] for r in pts], color=color, lw=2,
            marker="o", ms=6, mec=SURFACE, mew=1.5, label=label, zorder=2)
    for r in pts:  # concurrency (and the block size picked for it) beside each point
        text = f"B={r['batch']}" if key == "ar" else f"B={r['batch']} · blk {r['block']}"
        ax.annotate(text, (r["tps_per_user"], r["throughput"]), xytext=(5, 3),
                    textcoords="offset points", fontsize=7, color=MUTED)

ax.set_xlabel("Per-user decode speed (tokens/s per request)", color=INK)
ax.set_ylabel("Total throughput (tokens/s)", color=INK)
ds = rows[0]["dataset"]
ax.set_title(f"Qwen3-8B on Ryzen AI 9 HX 370: throughput vs per-user speed\n"
             f"one point per concurrency B (static batch), DFlash at its best block size, {ds}, {rows[0]['max_new_tokens']} new tokens, greedy",
             fontsize=10, color=INK, loc="left")
ax.grid(color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=8)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.legend(fontsize=7.5, frameon=False, labelcolor=INK, loc="upper right")
fig.tight_layout()
fig.savefig(dst, facecolor=SURFACE)
print("wrote", dst)
