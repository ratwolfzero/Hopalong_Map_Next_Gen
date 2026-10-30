"""Henon-Heiles orbit density with velocity Verlet (leapfrog): two-pass checkpointed, minimal version.
Pass 1 walks the orbit once (serial): extents + full state at the start of every segment.
Pass 2 replays the segments in parallel, each into its own histogram. Needs: numpy, numba, matplotlib."""
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from numba import njit, prange, get_num_threads

DT, SHAPE = 0.01, (1000, 1000)                       # time step, histogram size (rows, columns)
Y0 = -0.1                                            # start: x = 0, py = 0, px from energy E = 1/8
START = (0.0, Y0, float(np.sqrt(2 * (0.125 - (0.5 * Y0 ** 2 - Y0 ** 3 / 3)))), 0.0)


@njit(inline='always')
def force(x, y):                                     # F = -grad V,  V = (x^2 + y^2)/2 + x^2*y - y^3/3
    return -x - 2.0 * x * y, -y - x * x + y * y


@njit(inline='always')
def step(x, y, px, py, fx, fy, h):                   # velocity Verlet: half kick, drift, new force, half kick
    px += 0.5 * h * fx
    py += 0.5 * h * fy
    x += h * px
    y += h * py
    fx, fy = force(x, y)
    px += 0.5 * h * fx
    py += 0.5 * h * fy
    return x, y, px, py, fx, fy


@njit
def first_pass(x, y, px, py, h, tasks, seg, rem):    # serial
    cps = np.empty((tasks, 4))
    fx, fy = force(x, y)
    min_x = max_x = x
    min_y = max_y = y
    for k in range(tasks):
        cps[k, 0], cps[k, 1], cps[k, 2], cps[k, 3] = x, y, px, py      # checkpoint = the full state
        for _ in range(seg + rem if k == tasks - 1 else seg):
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            x, y, px, py, fx, fy = step(x, y, px, py, fx, fy, h)
    return (min_x, max_x, min_y, max_y), cps


@njit(parallel=True)
def second_pass(h, tasks, seg, rem, cps, ext, shape):    # parallel: one private histogram per task
    rows, cols = shape
    sx, sy = cols / (ext[1] - ext[0]), rows / (ext[3] - ext[2])
    hists = np.zeros((tasks, rows, cols), dtype=np.uint32)
    for k in prange(tasks):
        x, y, px, py = cps[k, 0], cps[k, 1], cps[k, 2], cps[k, 3]
        fx, fy = force(x, y)                         # pure function of (x, y): same bits as in pass 1
        for _ in range(seg + rem if k == tasks - 1 else seg):
            i = min(int((x - ext[0]) * sx), cols - 1)
            j = min(int((y - ext[2]) * sy), rows - 1)
            hists[k, j, i] += 1
            x, y, px, py, fx, fy = step(x, y, px, py, fx, fy, h)
    return hists


def density(n, tasks):
    seg, rem = divmod(n, tasks)                      # the last segment also takes the rem leftover points
    t0 = time.perf_counter()
    ext, cps = first_pass(*START, DT, tasks, seg, rem)
    t1 = time.perf_counter()
    img = second_pass(DT, tasks, seg, rem, cps, ext, SHAPE).sum(axis=0, dtype=np.uint64)
    return img, ext, t1 - t0, time.perf_counter() - t1


def plot(img, ext, n):                                # hits per pixel, sqrt colour scale (lifts the sparse regions)
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='gainsboro')
    im = ax.imshow(img, origin='lower', extent=ext, cmap='hot', interpolation='none',
                   norm=PowerNorm(gamma=0.5, vmin=0, vmax=int(img.max())))
    ax.set_title(f'Henon-Heiles orbit, velocity Verlet, E = 1/8, n = {n:_}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(im, ax=ax, location='bottom', label='hits per pixel (sqrt colour scale)')
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    n = 100_000_003                                  # deliberately not divisible by the segment count
    density(1000, 2)                                 # compile once, so the timings below are fair
    ref, ref_ext, t1, t2 = density(n, 1)             # 1 segment = the plain serial two-pass loop
    print(f'1 segment (plain serial loop):  pass 1 {t1:.2f} s, pass 2 {t2:.2f} s, total {t1 + t2:.2f} s')
    tasks = 2 * get_num_threads()
    img, ext, t1, t2 = density(n, tasks)
    print(f'{tasks} segments, {get_num_threads()} threads:      pass 1 {t1:.2f} s, pass 2 {t2:.2f} s, total {t1 + t2:.2f} s')
    print('identical to the serial loop (every pixel and every extent):', np.array_equal(img, ref) and ext == ref_ext)
    plot(img, ext, n)
    plt.show()
