#!/usr/bin/env python3
"""
bench_readme.py - reproduces every number in the README section "Verification and benchmarks"

Options:
    --n-speed N   points for the speed benchmark            (default 100_000_000)
    --reps R      repetitions, the median is reported       (default 3)
    --mem-n N     points for the memory comparison          (default 20_000_000)
    --big         also run the memory comparison at 100 million points. The numpy route then
                  needs roughly 6 GB of RAM; it is skipped automatically if that does not fit.
    --out FILE    where to write the README text            (default verification_and_benchmarks.md)

Do not run other heavy programs while it runs: timings and peak memory would be disturbed.
"""
import argparse
import contextlib
import io
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("MPLBACKEND", "Agg")  # never open windows while benchmarking
import numpy as np
from math import copysign, sqrt, fabs
from numba import njit

SIZE = (1000, 1000)  # rows, columns - same as the script's default image size


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------

def plural(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


def _sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def cpu_name():
    name = ""
    if sys.platform == "darwin":
        name = _sh(["sysctl", "-n", "machdep.cpu.brand_string"])
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        name = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    return name or platform.processor() or platform.machine()


def ram_gb():
    try:
        if sys.platform == "darwin":
            return int(_sh(["sysctl", "-n", "hw.memsize"])) / 2**30
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except Exception:
        return None


def fmt_mb(mb):
    return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb:.0f} MB"


@njit(cache=True)
def orbit_points(a, b, c, n):
    # The "natural" numpy route: materialise all n points (x_0 ... x_{n-1}) in two arrays.
    xs = np.empty(n)
    ys = np.empty(n)
    x = 0.0
    y = 0.0
    for i in range(n):
        xs[i] = x
        ys[i] = y
        x, y = y - copysign(1.0, x) * sqrt(fabs(b * x - c)), a - x
    return xs, ys


# ----------------------------------------------------------------------------------------------
# child processes for the memory comparison (peak RSS must be measured per process)
# ----------------------------------------------------------------------------------------------

def child_ours(n):
    import numba
    import hopalong_fast as hf
    hf.warm_up()
    print("READY", flush=True)          # the parent now reads our baseline memory ...
    sys.stdin.readline()                # ... and sends GO
    t = time.perf_counter()
    counts, ext, _ = hf.compute_trajectory_image(0.4, 1.0, 0.0, n, SIZE, uniform_bins=True)
    _ = counts * hf.density_factor(counts, ext)
    dt = time.perf_counter() - t
    print(json.dumps({"seconds": dt, "threads": numba.get_num_threads(), "tasks": hf.make_plan(n)[0]}),
          flush=True)


def child_numpy(n):
    orbit_points(0.4, 1.0, 0.0, 10)     # compile first
    print("READY", flush=True)
    sys.stdin.readline()
    t = time.perf_counter()
    xs, ys = orbit_points(0.4, 1.0, 0.0, n)
    _ = np.histogram2d(xs, ys, bins=SIZE, density=True)
    print(json.dumps({"seconds": time.perf_counter() - t}), flush=True)


def rss_mb(pid):
    out = _sh(["ps", "-o", "rss=", "-p", str(pid)])  # resident set size in KB, macOS and Linux
    return int(out) / 1024 if out.isdigit() else None


def run_child(mode, n):
    # The parent samples the child's resident memory every 10 ms while it computes. (The process-wide
    # "peak" counter of the OS is no good here: the import of numba/matplotlib alone already sets a
    # high-water mark that can hide the memory of the computation.)
    err = tempfile.TemporaryFile(mode="w+")
    p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--child", mode, str(n)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True)
    try:
        line = p.stdout.readline()
        while line and line.strip() != "READY":
            line = p.stdout.readline()
        if not line:
            p.wait()
            err.seek(0)
            return {"failed": "child exited before it was ready: " + (err.read().strip().splitlines() or ["?"])[-1]}
        base = rss_mb(p.pid)
        p.stdin.write("GO\n")
        p.stdin.flush()
        peak = base
        while p.poll() is None:
            r = rss_mb(p.pid)
            if r is not None and (peak is None or r > peak):
                peak = r
            time.sleep(0.01)
        out = p.stdout.read().strip().splitlines()
    finally:
        if p.poll() is None:
            p.kill()
    if p.returncode != 0:
        err.seek(0)
        last = (err.read().strip().splitlines() or ["?"])[-1]
        if p.returncode < 0:
            return {"failed": f"process was killed (signal {-p.returncode}), probably out of memory"}
        return {"failed": f"crashed: {last}"}
    res = json.loads(out[-1])
    res["extra_mb"] = max(0.0, (peak or 0) - (base or 0))
    return res


# ----------------------------------------------------------------------------------------------
# 1) verification
# ----------------------------------------------------------------------------------------------

COUNT_CASES = [(0.4, 1.0, 0.0, 5000), (-3.14, 0.3, 0.3, 1_000_037),
               (2.0, 1.0, 1.0, 3_000_001), (7.16878197155893, 8.439731, 2.2, 2_345_678)]
DENSITY_CASES = [(0.4, 1.0, 0.0, 2_000_003), (-3.14, 0.3, 0.3, 3_000_001),
                 (7.16878197155893, 8.439731, 2.2, 2_345_678)]


def check_counts_vs_original(hf, basic):
    ok = True
    for a, b, c, n in COUNT_CASES:
        ext_old = basic.compute_trajectory_extents(a, b, c, n)
        img_old = basic.compute_trajectory_and_image(a, b, c, n, ext_old, SIZE)
        img_new, ext_new, _ = hf.compute_trajectory_image(a, b, c, n, SIZE)
        same = tuple(ext_old) == tuple(ext_new) and np.array_equal(img_old, img_new)
        ok &= same
        print(f"   a={a} b={b} c={c} n={n:,}: {'identical' if same else 'DIFFERENT'}")
    return ok


def check_density_vs_numpy(hf):
    cells_bad, cells_total, worst_rel, worst_integral_err = 0, 0, 0.0, 0.0
    for a, b, c, n in DENSITY_CASES:
        xs, ys = orbit_points(a, b, c, n)
        rng = ((xs.min(), xs.max()), (ys.min(), ys.max()))
        ref_counts, xe, ye = np.histogram2d(xs, ys, bins=SIZE, range=rng)
        ref_dens, _, _ = np.histogram2d(xs, ys, bins=SIZE, range=rng, density=True)
        counts, ext, _ = hf.compute_trajectory_image(a, b, c, n, SIZE, uniform_bins=True)
        dens = counts * hf.density_factor(counts, ext)
        bad = int((counts.T != ref_counts).sum())          # numpy returns [x, y]; ours [y, x]
        rel = float(np.abs(dens.T - ref_dens).max() / ref_dens.max())
        integral = float(dens.sum() * (xe[1] - xe[0]) * (ye[1] - ye[0]))
        cells_bad += bad
        cells_total += counts.size
        worst_rel = max(worst_rel, rel)
        worst_integral_err = max(worst_integral_err, abs(integral - 1.0))
        print(f"   a={a} b={b} c={c} n={n:,}: cells differing {bad}, "
              f"max relative density difference {rel:.1e}, integral {integral:.9f}")
    return cells_bad, cells_total, worst_rel, worst_integral_err


def check_colour_scales(hf):
    import matplotlib.pyplot as plt
    image, _, _ = hf.compute_trajectory_image(0.4, 1.0, 0.0, 2_000_000, SIZE)
    m = float(image.max())
    levels = {"linear": lambda d: d / m, "sqrt": lambda d: np.sqrt(d / m),
              "log": lambda d: np.log1p(d) / np.log1p(m)}
    worst = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        for scale, f in levels.items():
            params = dict(a=0.4, b=1.0, c=0.0, n=2_000_000, scale=scale, output="counts")
            with contextlib.redirect_stdout(io.StringIO()):
                path = hf.save_png(image, params, "gray", hf.make_norm(image, scale), directory=tmp)
            grey = plt.imread(path)[..., 0]
            expected = np.flipud(f(image.astype(float)))      # PNG row 0 = top = highest y
            err = float(np.abs(grey - expected).max() * 255)
            worst = max(worst, err)
            print(f"   {scale}: max deviation {err:.2f} grey levels (of 255)")
    return worst


# ----------------------------------------------------------------------------------------------
# 2) speed
# ----------------------------------------------------------------------------------------------

def bench_speed(hf, basic, n, reps):
    import numba
    a, b, c = 0.4, 1.0, 0.0
    max_threads = numba.config.NUMBA_NUM_THREADS
    configs = [1] + ([max_threads] if max_threads > 1 else [])
    plain, fast = [], {k: [] for k in configs}
    for r in range(reps):
        t0 = time.perf_counter()
        ext = basic.compute_trajectory_extents(a, b, c, n)
        t1 = time.perf_counter()
        ref = basic.compute_trajectory_and_image(a, b, c, n, ext, SIZE)
        t2 = time.perf_counter()
        plain.append((t1 - t0, t2 - t1))
        for k in configs:
            numba.set_num_threads(k)
            img, _, (p1, p2) = hf.compute_trajectory_image(a, b, c, n, SIZE)
            fast[k].append((p1, p2))
            assert np.array_equal(ref, img), "image differs from the original!"
        print(f"   repetition {r + 1}/{reps} done")
    numba.set_num_threads(max_threads)

    def med(rows):
        p1 = statistics.median(x[0] for x in rows)
        p2 = statistics.median(x[1] for x in rows)
        return p1, p2, p1 + p2
    return med(plain), {k: med(v) for k, v in fast.items()}, max_threads


# ----------------------------------------------------------------------------------------------
# 3) memory
# ----------------------------------------------------------------------------------------------

def bench_memory(n, label):
    ours = run_child("ours", n)
    ram = ram_gb()
    need_gb = n * 60 / 2**30  # measured: about 55 bytes per point for the numpy route
    if ram is not None and need_gb > 0.6 * ram:
        npy = {"skipped": f"skipped: would need about {need_gb:.0f} GB, machine has {ram:.0f} GB"}
    else:
        npy = run_child("numpy", n)
    print(f"   {label}: two-pass {ours} | numpy {npy}")
    return ours, npy


def cell(res):
    if "failed" in res:
        return "ran out of memory (process killed)"
    if "skipped" in res:
        return res["skipped"]
    return fmt_mb(res["extra_mb"])


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-speed", type=float, default=100_000_000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--mem-n", type=float, default=20_000_000)
    ap.add_argument("--big", action="store_true")
    ap.add_argument("--out", default="verification_and_benchmarks.md")
    ap.add_argument("--child", nargs=2, metavar=("MODE", "N"))
    args = ap.parse_args()

    if args.child:  # internal: run one memory measurement and print JSON
        mode, n = args.child[0], int(float(args.child[1]))
        (child_ours if mode == "ours" else child_numpy)(n)
        return
    if args.quick:
        args.n_speed, args.reps, args.mem_n = 10_000_000, 1, 5_000_000

    import numba
    import hopalong_basic as basic
    import hopalong_fast as hf
    hf.warm_up()
    n_speed, mem_n = int(args.n_speed), int(args.mem_n)

    hw = (f"{cpu_name()}, {plural(numba.config.NUMBA_NUM_THREADS, 'thread')}, "
          f"{ram_gb() or 0:.0f} GB RAM, {platform.system()} {platform.release()}, "
          f"Python {platform.python_version()}, numpy {np.__version__}, numba {numba.__version__}")
    print("Machine:", hw, "\n")

    print("1/4  counts mode vs. the original script")
    ok_counts = check_counts_vs_original(hf, basic)
    print("2/4  density mode vs. numpy.histogram2d")
    bad, total, rel, int_err = check_density_vs_numpy(hf)
    print("     colour scales (PNG vs. formulas)")
    grey = check_colour_scales(hf)
    print(f"3/4  speed, {n_speed:,} points, median of {args.reps}")
    plain, fast, max_threads = bench_speed(hf, basic, n_speed, args.reps)
    print(f"4/4  memory, {mem_n:,} points" + (" and 100,000,000 points" if args.big else ""))
    mem = {mem_n: bench_memory(mem_n, f"{mem_n:,}")}
    if args.big:
        mem[100_000_000] = bench_memory(100_000_000, "100,000,000")
    threads_mem = mem[mem_n][0].get("threads", "?")
    tasks_mem = mem[mem_n][0].get("tasks", "?")

    # ---- build the README text --------------------------------------------------------------
    L = []
    L.append("## Verification and benchmarks\n")
    L.append(f"Measured on: {hw}.\n")
    L.append("Checks that were run during development:\n")
    L.append(f"- **Counts mode vs. the original simple two-pass script:** the extents and the complete "
             f"1000 × 1000 image are **{'bit-for-bit identical' if ok_counts else 'NOT identical (!)'}** "
             f"for {len(COUNT_CASES)} different parameter sets, including values of `n` that do not "
             f"divide evenly into segments and runs with several tasks.")
    cells = ("**no cell differs** in its count" if bad == 0
             else f"**{bad} of {total:,} cells differ** in their count")
    L.append(f"- **Density mode vs. `numpy.histogram2d(..., density=True)`:** with 2 to 3 million points on "
             f"{len(DENSITY_CASES)} parameter sets, {cells}, the densities agree to floating-point rounding "
             f"(relative difference about {rel:.0e}), and the integral is within {int_err:.0e} of 1.")
    L.append(f"- **Colour scales:** the saved PNGs were compared with the three formulas above "
             f"(using a grey colour map) and agree to within {max(1, int(np.ceil(grey)))} grey level"
             f"{'s' if np.ceil(grey) > 1 else ''} out of 255, which is the rounding of a 256-entry colour "
             f"map and 8-bit PNG; the orientation (y pointing up) was checked at the same time.\n")

    L.append(f"Speed, {n_speed:,} points, parameters a = 0.4, b = 1.0, c = 0.0, 1000 × 1000 grid, "
             f"median of {plural(args.reps, 'run')}. The plain loop always runs on one core, but its speed still "
             f"depends on the CPU, so it was measured on this machine too:\n")
    L.append("| | Pass 1 | Pass 2 | Total |")
    L.append("| --- | --- | --- | --- |")
    L.append("| Plain two-pass numba loop (1 core by nature) | " + " | ".join(f"{v:.2f} s" for v in plain) + " |")
    for k in sorted(fast):
        name = f"`hopalong_fast.py`, {k} thread" + ("s" if k > 1 else "")
        L.append(f"| {name} | " + " | ".join(f"{v:.2f} s" for v in fast[k]) + " |")
    L.append("")

    L.append(f"Memory, density histogram of the same orbit (extra peak RAM on top of what the interpreter "
             f"and the libraries already use; `hopalong_fast.py` with {plural(threads_mem, 'thread')} = {plural(tasks_mem, 'task')}, "
             f"so at most {tasks_mem} × 4 MB of tally sheets plus the 8 MB result; the operating system only counts the parts that were actually touched):\n")
    heads = [f"{n:,} points" for n in mem]
    L.append("| | Extra peak RAM, " + " | ".join(heads) + " |")
    L.append("| --- | " + " | ".join("---" for _ in mem) + " |")
    L.append("| Generate the points into arrays, then `numpy.histogram2d` | " +
             " | ".join(cell(mem[n][1]) for n in mem) + " |")
    L.append("| `hopalong_fast.py` | " + " | ".join(cell(mem[n][0]) for n in mem) + " |")
    text = "\n".join(L) + "\n"

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + "=" * 100 + f"\nREADME text written to {args.out}:\n" + "=" * 100 + "\n" + text)


if __name__ == "__main__":
    main()
