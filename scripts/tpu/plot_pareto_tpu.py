"""Plot TPU throughput vs per-user TPS (one curve per system), optionally with the Ryzen AI curves.

Log axes, since TPU and Ryzen numbers differ by an order of magnitude.

Usage: python scripts/tpu/plot_pareto_tpu.py [docs/data/pareto_tpu.jsonl] [docs/pareto_tpu.png] [--with-ryzen docs/data/pareto.jsonl]
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

argv = sys.argv[1:]
ryzen = None
if "--with-ryzen" in argv:
    i = argv.index("--with-ryzen")
    ryzen = argv[i + 1]
    del argv[i:i + 2]
src = argv[0] if argv else "docs/data/pareto_tpu.jsonl"
dst = argv[1] if len(argv) > 1 else "docs/pareto_tpu.png"
rows = [json.loads(line) for line in open(src)]
if ryzen:
    rows += [json.loads(line) for line in open(ryzen)]

chip = next(r["chip"] for r in rows if "chip" in r)
SYSTEMS = [
    ("tpu_ar", f"AR ({chip}, bf16)", "#2a78d6", "-"),
    ("tpu_dflash", f"DFlash ({chip}, bf16)", "#eb6834", "-"),
    ("ar", "AR (Ryzen 890M GPU, W8A8)", "#2a78d6", ":"),
    ("gpu", "DFlash (Ryzen 890M GPU, W8A8)", "#eb6834", ":"),
    ("hetero", "DFlash (Ryzen GPU + NPU)", "#1baf7a", ":"),
]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def best_per_batch(points):
    """One point per concurrency: the block size with the highest throughput at that batch."""
    best = {}
    for r in points:
        if r["batch"] not in best or r["throughput"] > best[r["batch"]]["throughput"]:
            best[r["batch"]] = r
    return [best[b] for b in sorted(best)]


fig, ax = plt.subplots(figsize=(8, 5.2), dpi=160)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
for key, label, color, ls in SYSTEMS:
    pts = best_per_batch([r for r in rows if r["system"] == key])
    if not pts:
        continue
    ax.plot([r["tps_per_user"] for r in pts], [r["throughput"] for r in pts], color=color, lw=2, ls=ls,
            marker="o", ms=6, mec=SURFACE, mew=1.5, label=label, zorder=2)
    for r in pts:
        text = f"B={r['batch']}" if r["block"] == 1 else f"B={r['batch']} · blk {r['block']}"
        ax.annotate(text, (r["tps_per_user"], r["throughput"]), xytext=(5, 3),
                    textcoords="offset points", fontsize=7, color=MUTED)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Per-user decode speed (tokens/s per request)", color=INK)
ax.set_ylabel("Total throughput (tokens/s)", color=INK)
ax.set_title(f"Qwen3-8B DFlash on {chip}: throughput vs per-user speed\n"
             f"one point per concurrency B (static batch), DFlash at its best block size, "
             f"{rows[0]['dataset']}, {rows[0]['max_new_tokens']} new tokens, greedy",
             fontsize=10, color=INK, loc="left")
ax.grid(color=GRID, lw=0.8, which="both")
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=8)
ax.legend(fontsize=7.5, frameon=False, labelcolor=INK, loc="lower right")
fig.tight_layout()
fig.savefig(dst, facecolor=SURFACE)
print("wrote", dst)
