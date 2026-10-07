"""Plot throughput vs per-user TPS from pareto.py results, with the Pareto frontier.

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

# Color = system (fixed categorical order), marker = draft block size.
SYSTEMS = [
    ("ar", "AR (GPU W8A8)", "#2a78d6"),
    ("gpu", "DFlash, all GPU", "#eb6834"),
    ("hetero", "DFlash, GPU target + NPU draft", "#1baf7a"),
]
MARKERS = {1: "o", 16: "o", 8: "s", 4: "^"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def frontier(points):
    """Points not dominated in (tps_per_user, throughput), sorted by TPS."""
    best = []
    for r in sorted(points, key=lambda r: (-r["tps_per_user"], -r["throughput"])):
        if not best or r["throughput"] > best[-1]["throughput"]:
            best.append(r)
    return best


fig, ax = plt.subplots(figsize=(8, 5.2), dpi=160)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
front = frontier(rows)
ax.plot([r["tps_per_user"] for r in front], [r["throughput"] for r in front],
        color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1, label="Pareto frontier")

for key, label, color in SYSTEMS:
    sys_rows = [r for r in rows if r["system"] == key]
    for block in sorted({r["block"] for r in sys_rows}, reverse=True):
        pts = sorted((r for r in sys_rows if r["block"] == block), key=lambda r: r["batch"])
        name = label if key == "ar" else f"{label}, block {block}"
        ax.plot([r["tps_per_user"] for r in pts], [r["throughput"] for r in pts],
                color=color, lw=2, alpha=1.0 if block in (1, 16) else 0.55, zorder=2,
                marker=MARKERS.get(block, "o"), ms=6, mec=SURFACE, mew=1.5, label=name)
        for r in pts:  # batch size beside each point, in text ink
            ax.annotate(f"{r['batch']}", (r["tps_per_user"], r["throughput"]), xytext=(4, 4),
                        textcoords="offset points", fontsize=7, color=MUTED)

ax.set_xlabel("Per-user decode speed (tokens/s per request)", color=INK)
ax.set_ylabel("Total throughput (tokens/s)", color=INK)
ds = rows[0]["dataset"]
ax.set_title(f"Qwen3-8B on Ryzen AI 9 HX 370: throughput vs per-user speed\n"
             f"static batching (numbers = batch size), {ds}, {rows[0]['max_new_tokens']} new tokens, greedy",
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
print("wrote", dst, "| frontier:", [(r["system"], r["block"], r["batch"]) for r in front])
