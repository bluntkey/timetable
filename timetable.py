#!/usr/bin/env python3
"""
Exam Timetable Scheduling  --  Graph colouring with room (hall) capacity.

Given N courses, a conflict graph (two courses conflict if some student takes both)
and a hall limit R (max exams per slot), assign every course to a time slot so that
  * conflicting courses never share a slot,
  * no slot holds more than R exams,
and the number of slots is minimised.

Algorithms
  1. Welsh-Powell  (greedy, capacity-aware)
  2. DSatur        (greedy, capacity-aware)
  3. Exact         (DSatur-ordered backtracking / branch-and-bound, with
                    clique + ceil(N/R) lower bounds, symmetry breaking and
                    forward checking; seeded with the greedy upper bound)

Usage examples
  python timetable.py --input sample_input.txt
  python timetable.py --random 40 6 --density 0.25 --seed 1
  python timetable.py                       # interactive prompt
Courses are numbered 1..N in the input and in all output.
"""
import argparse
import csv
import math
import os
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
import networkx as nx

sys.setrecursionlimit(20000)


# ----------------------------------------------------------------------------
# Instance handling
# ----------------------------------------------------------------------------
class Instance:
    def __init__(self, n, edges, R):
        self.n, self.R = n, R
        self.adj = [set() for _ in range(n)]
        for u, v in edges:
            if u != v:
                self.adj[u].add(v)
                self.adj[v].add(u)
        self.edges = sorted({(min(u, v), max(u, v)) for u, v in edges if u != v})
        self.deg = [len(a) for a in self.adj]

    @property
    def density(self):
        return 0.0 if self.n < 2 else len(self.edges) / (self.n * (self.n - 1) / 2)


def read_instance(path):
    """Option 1 file format:  'N M R' then M lines 'u v'  (1-indexed; 0-indexed is auto-detected)."""
    with open(path) as f:
        lines = [l.split("#")[0].strip() for l in f]
    lines = [l for l in lines if l]
    n, m, R = map(int, lines[0].split()[:3])
    raw = [tuple(map(int, l.split()[:2])) for l in lines[1:1 + m]]
    shift = 0 if any(0 in e for e in raw) else 1  # 0 present -> file is 0-indexed
    edges = []
    for u, v in raw:
        u, v = u - shift, v - shift
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"Edge ({u + shift}, {v + shift}) is outside 1..{n}")
        edges.append((u, v))
    return Instance(n, edges, R)


def random_instance(n, R, density, seed=None):
    """Option 2: Erdos-Renyi conflict graph G(n, density)."""
    rng = random.Random(seed)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < density]
    return Instance(n, edges, R)


# ----------------------------------------------------------------------------
# Result container + validation
# ----------------------------------------------------------------------------
class Result:
    def __init__(self, name, slot_of, order, elapsed, proven, note=""):
        self.name = name
        self.slot_of = slot_of          # slot index (0-based) per course
        self.order = order              # [(course, slot)] in assignment order
        self.elapsed = elapsed
        self.proven = proven
        self.note = note
        self.slots_used = len(set(slot_of))

    def timetable(self):
        tt = [[] for _ in range(self.slots_used)]
        for v, s in enumerate(self.slot_of):
            tt[s].append(v)
        return tt


def validate(inst, slot_of):
    assert all(s >= 0 for s in slot_of), "unassigned course"
    for u, v in inst.edges:
        assert slot_of[u] != slot_of[v], f"conflict {u + 1}-{v + 1} share slot {slot_of[u] + 1}"
    cnt = {}
    for s in slot_of:
        cnt[s] = cnt.get(s, 0) + 1
    assert max(cnt.values()) <= inst.R, "hall capacity exceeded"
    assert sorted(cnt) == list(range(len(cnt))), "slot numbering has gaps"


# ----------------------------------------------------------------------------
# Lower bounds
# ----------------------------------------------------------------------------
def greedy_clique(inst, tries=60):
    """Heuristic large clique: any clique of size w needs w distinct slots."""
    n, adj = inst.n, inst.adj
    if n == 0:
        return []
    starts = sorted(range(n), key=lambda v: -inst.deg[v])[:tries]
    best = []
    for s in starts:
        clique, cand = [s], set(adj[s])
        while cand:
            v = max(cand, key=lambda x: len(adj[x] & cand))
            clique.append(v)
            cand &= adj[v]
        if len(clique) > len(best):
            best = clique
    return best


def lower_bound(inst, clique):
    return max(math.ceil(inst.n / inst.R), len(clique), 1 if inst.n else 0)


# ----------------------------------------------------------------------------
# Greedy 1: Welsh-Powell (capacity aware)
# ----------------------------------------------------------------------------
def welsh_powell(inst):
    t0 = time.perf_counter()
    n, R, adj = inst.n, inst.R, inst.adj
    order_v = sorted(range(n), key=lambda v: (-inst.deg[v], v))
    slot_of = [-1] * n
    order, s, left = [], 0, n
    while left:
        members = set()
        for v in order_v:
            if slot_of[v] == -1 and len(members) < R and adj[v].isdisjoint(members):
                slot_of[v] = s
                members.add(v)
                order.append((v, s))
                left -= 1
        s += 1
    return Result("Welsh-Powell", slot_of, order, time.perf_counter() - t0, False)


# ----------------------------------------------------------------------------
# Greedy 2: DSatur (capacity aware)
# ----------------------------------------------------------------------------
def dsatur(inst):
    t0 = time.perf_counter()
    n, R, adj = inst.n, inst.R, inst.adj
    slot_of = [-1] * n
    nbr_slots = [set() for _ in range(n)]
    size, order = [], []
    uncolored = set(range(n))
    while uncolored:
        v = max(uncolored, key=lambda x: (len(nbr_slots[x]), inst.deg[x], -x))
        s = 0
        while s in nbr_slots[v] or (s < len(size) and size[s] >= R):
            s += 1
        if s == len(size):
            size.append(0)
        slot_of[v] = s
        size[s] += 1
        order.append((v, s))
        for u in adj[v]:
            nbr_slots[u].add(s)
        uncolored.discard(v)
    return Result("DSatur", slot_of, order, time.perf_counter() - t0, False)


# ----------------------------------------------------------------------------
# Exact: backtracking / branch-and-bound
# ----------------------------------------------------------------------------
class _Timeout(Exception):
    pass


def _feasible(inst, k, clique, deadline):
    """Decision problem: can all courses fit in k slots?  Returns (status, slot_of, trail)."""
    n, R, adj, deg = inst.n, inst.R, inst.adj, inst.deg
    slot = [-1] * n
    size = [0] * k
    cnt = [[0] * k for _ in range(n)]   # cnt[v][s] = #neighbours of v already in slot s
    trail = []
    nodes = [0]

    def assign(v, s):
        slot[v] = s
        size[s] += 1
        trail.append((v, s))
        for u in adj[v]:
            cnt[u][s] += 1

    def unassign(v, s):
        slot[v] = -1
        size[s] -= 1
        trail.pop()
        for u in adj[v]:
            cnt[u][s] -= 1

    if len(clique) > k:
        return "unsat", None, None
    for i, v in enumerate(clique):     # symmetry breaking: clique takes slots 0..w-1
        assign(v, i)
    unc = set(range(n)) - set(clique)

    def rec(used):
        if not unc:
            return True
        nodes[0] += 1
        if nodes[0] & 255 == 0 and time.perf_counter() > deadline:
            raise _Timeout
        lim = min(used + 1, k)          # may open at most one new (empty) slot
        best, bdom = None, None
        for v in unc:                   # forward checking + smallest-domain (DSatur-like) choice
            c = cnt[v]
            dom = [s for s in range(lim) if c[s] == 0 and size[s] < R]
            if not dom:
                return False
            if best is None or len(dom) < len(bdom) or (len(dom) == len(bdom) and deg[v] > deg[best]):
                best, bdom = v, dom
        unc.discard(best)
        for s in bdom:
            assign(best, s)
            if rec(max(used, s + 1)):
                return True
            unassign(best, s)
        unc.add(best)
        return False

    try:
        ok = rec(len(clique))
    except _Timeout:
        return "timeout", None, None
    return ("sat", slot[:], trail[:]) if ok else ("unsat", None, None)


def exact_bb(inst, incumbent, time_limit):
    """Tighten the greedy upper bound one slot at a time until infeasibility is proven."""
    t0 = time.perf_counter()
    deadline = t0 + time_limit
    clique = greedy_clique(inst)
    lb = lower_bound(inst, clique)
    best_slot, best_order = incumbent.slot_of[:], incumbent.order[:]
    best_k = incumbent.slots_used
    proven, note = False, f"LB={lb}"
    while True:
        if best_k <= lb:
            proven = True
            break
        status, sol, trail = _feasible(inst, best_k - 1, clique, deadline)
        if status == "sat":
            best_slot, best_order, best_k = sol, trail, len(set(sol))
        elif status == "unsat":
            proven = True
            break
        else:
            note += ", time limit hit"
            break
    return Result("Exact B&B", best_slot, best_order, time.perf_counter() - t0, proven, note)


# ----------------------------------------------------------------------------
# Output: text + CSV
# ----------------------------------------------------------------------------
def print_summary(inst, results, lb):
    print(f"\nInstance: N={inst.n}  conflicts={len(inst.edges)}  density={inst.density:.3f}  R={inst.R}")
    print(f"Lower bound on slots: {lb}  (max of ceil(N/R)={math.ceil(inst.n / inst.R)} and clique size)\n")
    print(f"{'Algorithm':<15}{'Slots':>6}  {'Proven optimal?':<34}{'Time (s)':>10}")
    print("-" * 67)
    for r in results:
        if r.proven:
            status = "YES (exhaustive proof)" if r.name.startswith("Exact") and r.slots_used > lb else "YES (meets lower bound)"
        elif r.slots_used <= lb:
            status = "YES (meets lower bound)"
        elif r.name.startswith("Exact"):
            status = "NO (time limit reached)"
        else:
            status = "NO (heuristic, no proof)"
        print(f"{r.name:<15}{r.slots_used:>6}  {status:<34}{r.elapsed:>10.4f}")


def slot_lines(res):
    return [f"Slot {i + 1}: " + ", ".join(str(v + 1) for v in sorted(c)) for i, c in enumerate(res.timetable())]


def write_csv(res, path):
    tt = [sorted(c) for c in res.timetable()]
    halls = max(len(c) for c in tt)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Slot"] + [f"Hall {h + 1}" for h in range(halls)])
        for i, c in enumerate(tt):
            w.writerow([f"Slot {i + 1}"] + [v + 1 for v in c] + [""] * (halls - len(c)))


# ----------------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------------
def palette(K):
    if K <= 20:
        order = list(range(0, 20, 2)) + list(range(1, 20, 2))
        cm = plt.get_cmap("tab20")
        return [cm(order[i]) for i in range(K)]
    cm = plt.get_cmap("hsv")
    return [cm(i / K) for i in range(K)]


def _text_color(rgba):
    r, g, b = rgba[:3]
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else "white"


def layout(inst):
    G = nx.Graph()
    G.add_nodes_from(range(inst.n))
    G.add_edges_from(inst.edges)
    return nx.spring_layout(G, seed=7, k=1.6 / math.sqrt(max(inst.n, 1)), iterations=200)


def draw_graph(ax, inst, pos, slot_of, pal, current=None, title=None):
    ax.clear()
    segs = [[pos[u], pos[v]] for u, v in inst.edges]
    if segs:
        ax.add_collection(LineCollection(segs, colors="#9a9a9a", linewidths=0.7, alpha=0.55, zorder=1))
    size = max(60, min(700, 9000 / max(inst.n, 1)))
    fs = max(5, min(11, 130 / max(inst.n, 1) + 4))
    for v in range(inst.n):
        s = slot_of[v]
        face = pal[s] if s >= 0 else (0.88, 0.88, 0.88, 1)
        cur = (v == current)
        ax.scatter(*pos[v], s=size * (1.5 if cur else 1), c=[face], edgecolors="black" if cur else "#444444",
                   linewidths=2.8 if cur else 0.8, zorder=3)
        ax.text(*pos[v], str(v + 1), ha="center", va="center", fontsize=fs,
                color=_text_color(face), zorder=4)
    ax.autoscale_view()
    ax.margins(0.08)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11)


def draw_grid(ax, rows, K, cols, pal, title=None):
    """rows: list (len K) of lists of course ids; slot 1 at the top."""
    ax.clear()
    fs = max(5, min(12, 150 / max(K, cols, 1) + 3))
    for r in range(K):
        for c in range(cols):
            filled = c < len(rows[r])
            face = pal[r] if filled else "#f3f3f3"
            ax.add_patch(Rectangle((c, K - 1 - r), 1, 1, facecolor=face, edgecolor="white", linewidth=1.5))
            if filled:
                ax.text(c + 0.5, K - 1 - r + 0.5, str(rows[r][c] + 1), ha="center", va="center",
                        fontsize=fs, color=_text_color(face), fontweight="bold")
    ax.set_xlim(0, cols)
    ax.set_ylim(0, K)
    ax.set_xticks([c + 0.5 for c in range(cols)])
    ax.set_xticklabels([f"Hall {c + 1}" for c in range(cols)], fontsize=8)
    ax.xaxis.tick_top()
    ax.set_yticks([K - 1 - r + 0.5 for r in range(K)])
    ax.set_yticklabels([f"Slot {r + 1}" for r in range(K)], fontsize=8)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, fontsize=11, pad=22)


def save_graph_png(inst, res, pos, path):
    pal = palette(res.slots_used)
    fig, ax = plt.subplots(figsize=(9, 8))
    draw_graph(ax, inst, pos, res.slot_of, pal,
               title=f"Conflict graph coloured by exam slot  ({res.name}: {res.slots_used} slots)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=pal[i], markeredgecolor="#444", label=f"Slot {i + 1}")
               for i in range(res.slots_used)]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8,
              ncol=1 if res.slots_used <= 18 else 2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_grid_png(inst, res, path):
    K = res.slots_used
    tt = [sorted(c) for c in res.timetable()]
    cols = max(len(c) for c in tt)
    pal = palette(K)
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * cols + 2), max(3, 0.55 * K + 1.5)))
    draw_grid(ax, tt, K, cols, pal, title=f"Exam timetable  ({res.name}: {K} slots, R={inst.R})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_animation(inst, res, pos, path, fps=3):
    K = res.slots_used
    pal = palette(K)
    final = [sorted(c) for c in res.timetable()]
    cols = max(len(c) for c in final)
    order = res.order
    step = max(1, len(order) // 80)               # cap the GIF at ~80 frames
    frames = list(range(0, len(order), step)) + [len(order) - 1]

    fig, (axg, axt) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.25, 1]})

    def render(i):
        slot_of = [-1] * inst.n
        rows = [[] for _ in range(K)]
        for v, s in order[:i + 1]:
            slot_of[v] = s
            rows[s].append(v)
        v, s = order[i]
        draw_graph(axg, inst, pos, slot_of, pal, current=v,
                   title=f"{res.name}: step {i + 1}/{len(order)}  -  course {v + 1} -> Slot {s + 1}")
        draw_grid(axt, rows, K, cols, pal, title="Timetable being built")

    ani = FuncAnimation(fig, render, frames=frames, repeat=False)
    fig.tight_layout()
    ani.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Comparison across N
# ----------------------------------------------------------------------------
def run_all(inst, time_limit):
    wp = welsh_powell(inst)
    ds = dsatur(inst)
    for r in (wp, ds):
        validate(inst, r.slot_of)
    inc = min((wp, ds), key=lambda r: r.slots_used)
    ex = exact_bb(inst, inc, time_limit)
    validate(inst, ex.slot_of)
    return wp, ds, ex


def sweep(sizes, R, density, trials, time_limit, seed, outdir):
    names = ["Welsh-Powell", "DSatur", "Exact B&B"]
    rows = []
    print(f"\nScalability sweep: density={density}, R={R}, {trials} random graphs per N, "
          f"exact time limit {time_limit}s")
    print(f"{'N':>5} | " + " | ".join(f"{n:>22}" for n in names))
    for n in sizes:
        agg = {nm: [0.0, 0.0, 0] for nm in names}
        for t in range(trials):
            inst = random_instance(n, R, density, seed + 1000 * n + t)
            for r in run_all(inst, time_limit):
                a = agg[r.name]
                a[0] += r.slots_used
                a[1] += r.elapsed
                a[2] += int(r.proven or r.slots_used <= lower_bound(inst, greedy_clique(inst)))
        line = f"{n:>5} | "
        for nm in names:
            s, tm, pv = agg[nm]
            rows.append((n, nm, s / trials, tm / trials, pv / trials))
            line += f"{s / trials:>6.2f} sl {tm / trials:>8.4f}s opt {pv}/{trials} | "
        print(line.rstrip(" |"))
    with open(os.path.join(outdir, "sweep_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "algorithm", "avg_slots", "avg_time_s", "fraction_proven_optimal"])
        w.writerows(rows)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    style = {"Welsh-Powell": ("tab:orange", "o", "--"), "DSatur": ("tab:blue", "s", "-."),
             "Exact B&B": ("tab:green", "D", "-")}
    for nm in names:
        pts = [r for r in rows if r[1] == nm]
        c, m, ls = style[nm]
        a1.plot([p[0] for p in pts], [p[2] for p in pts], color=c, marker=m, ls=ls, label=nm)
        a2.plot([p[0] for p in pts], [max(p[3], 1e-6) for p in pts], color=c, marker=m, ls=ls, label=nm)
    a1.set(xlabel="N (courses)", ylabel="Avg. slots used", title="Solution quality")
    a2.set(xlabel="N (courses)", ylabel="Avg. running time (s, log scale)", yscale="log", title="Running time")
    a2.axhline(time_limit, color="gray", ls=":", lw=1)
    a2.text(sizes[0], time_limit * 1.1, "exact time limit", color="gray", fontsize=8)
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle(f"Algorithm comparison  (density={density}, R={R}, mean of {trials} instances)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "comparison_chart.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def prompt_instance():
    print("Choose input option:\n  1) read conflict graph from file\n  2) generate a random conflict graph")
    if input("Option [1/2]: ").strip() == "1":
        return read_instance(input("File path: ").strip())
    n, R = map(int, input("N R: ").split())
    d = float(input("Edge density (0-1): "))
    return random_instance(n, R, d, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="Option 1: file with 'N M R' then M edge lines")
    ap.add_argument("--random", nargs=2, type=int, metavar=("N", "R"), help="Option 2: random graph")
    ap.add_argument("--density", type=float, default=0.2, help="edge density for --random (default 0.2)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--time-limit", type=float, default=30.0, help="seconds for the exact method")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--no-animation", action="store_true")
    ap.add_argument("--no-sweep", action="store_true", help="skip the N-scaling comparison chart")
    ap.add_argument("--sweep-sizes", default="10,20,30,40,50,60")
    ap.add_argument("--sweep-trials", type=int, default=3)
    ap.add_argument("--sweep-time-limit", type=float, default=5.0)
    ap.add_argument("--sweep-R", type=int, default=None, help="R for the sweep (default: R of the instance)")
    ap.add_argument("--sweep-density", type=float, default=None, help="default: density of --random, else 0.2")
    args = ap.parse_args()

    if args.input:
        inst = read_instance(args.input)
    elif args.random:
        inst = random_instance(args.random[0], args.random[1], args.density, args.seed)
    else:
        inst = prompt_instance()

    os.makedirs(args.outdir, exist_ok=True)
    wp, ds, ex = run_all(inst, args.time_limit)
    lb = lower_bound(inst, greedy_clique(inst))
    print_summary(inst, [wp, ds, ex], lb)

    final = ex  # exact result is never worse than the greedy incumbent
    print(f"\nFinal timetable ({final.name}, {final.slots_used} slots, "
          f"{'proven optimal' if final.proven or final.slots_used <= lb else 'best found'}):")
    lines = slot_lines(final)
    print("\n".join(lines))
    with open(os.path.join(args.outdir, "timetable.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    write_csv(final, os.path.join(args.outdir, "timetable.csv"))

    pos = layout(inst)
    save_graph_png(inst, final, pos, os.path.join(args.outdir, "conflict_graph.png"))
    save_grid_png(inst, final, os.path.join(args.outdir, "timetable_grid.png"))
    if not args.no_animation:
        save_animation(inst, final, pos, os.path.join(args.outdir, "assignment_animation.gif"))
        # also show how the greedy DSatur built its schedule, for comparison
        save_animation(inst, ds, pos, os.path.join(args.outdir, "assignment_animation_dsatur.gif"))
    if not args.no_sweep:
        R = args.sweep_R or inst.R
        dens = args.sweep_density if args.sweep_density is not None else (
            args.density if args.random else 0.2)
        sizes = [int(x) for x in args.sweep_sizes.split(",")]
        sweep(sizes, R, dens, args.sweep_trials, args.sweep_time_limit, args.seed or 0, args.outdir)
    print(f"\nFiles written to ./{args.outdir}/")


if __name__ == "__main__":
    main()
