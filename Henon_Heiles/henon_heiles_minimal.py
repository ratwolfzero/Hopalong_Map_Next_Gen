"""Henon-Heiles orbit density with leapfrog: two-pass checkpointed, 4 lanes, fused multiply-add.
Pass 1 walks the orbit once (serial): extents + full state at the start of every segment.
Pass 2 replays the segments in parallel, each into its own histogram; every worker advances
4 segments in lock-step (lanes) so the CPU can overlap their waiting. Needs: numpy, numba, matplotlib."""
import time
import matplotlib.pyplot as plt
import numpy as np
from llvmlite import ir
from matplotlib.colors import PowerNorm
from numba import get_num_threads, njit, prange, types
from numba.extending import intrinsic

LANES = 4                                            # the kernel below is written out for exactly 4 lanes
DT, SHAPE = 0.01, (1000, 1000)                       # time step, histogram size (rows, columns)
Y0, ENERGY = -0.1, 0.125                             # start: x = 0, py = 0, px from the energy


@intrinsic
def fma(typingctx, a, b, c):                         # a * b + c with ONE rounding: faster, and the same bits on every machine
    def codegen(context, builder, signature, args):
        dt = ir.DoubleType()
        fn = builder.module.declare_intrinsic('llvm.fma', [dt], ir.FunctionType(dt, [dt, dt, dt]))
        return builder.call(fn, args)
    return types.float64(types.float64, types.float64, types.float64), codegen


@njit(inline='always')
def force(x, y):                                     # F = -grad V,  V = (x^2 + y^2)/2 + x^2*y - y^3/3
    return -fma(2.0 * y, x, x), fma(y, y, fma(-x, x, -y))      # = (-x - 2xy,  -y - x^2 + y^2)


@njit(inline='always')
def step(x, y, px, py, fx, fy, h):                   # leapfrog: kick, drift (momentum runs half a step behind)
    px = fma(h, fx, px)
    py = fma(h, fy, py)
    x = fma(h, px, x)
    y = fma(h, py, y)
    fx, fy = force(x, y)
    return x, y, px, py, fx, fy


def start_state(y0=Y0, energy=ENERGY, h=DT):
    px = float(np.sqrt(2 * (energy - (0.5 * y0 * y0 - y0 * y0 * y0 / 3))))    # from H = energy at x = 0, py = 0
    fx, fy = force(0.0, y0)
    return 0.0, y0, px - 0.5 * h * fx, 0.0 - 0.5 * h * fy                 # momentum half a step back


START = start_state()


@njit
def first_pass(x, y, px, py, h, segments, seg, rem):          # serial
    cps = np.empty((segments, 4))
    fx, fy = force(x, y)
    min_x = max_x = x
    min_y = max_y = y
    for k in range(segments):
        cps[k, 0], cps[k, 1], cps[k, 2], cps[k, 3] = x, y, px, py      # checkpoint = the full state
        for _ in range(seg + rem if k == segments - 1 else seg):
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            x, y, px, py, fx, fy = step(x, y, px, py, fx, fy, h)
    return (min_x, max_x, min_y, max_y), cps


@njit(inline='always')
def hit(x, y, x_min, y_min, sx, sy, hist):
    rows, cols = hist.shape
    i = min(int((x - x_min) * sx), cols - 1)
    j = min(int((y - y_min) * sy), rows - 1)
    hist[j, i] += 1


@njit(parallel=True)
def second_pass(h, tasks, seg, rem, cps, ext, shape):          # parallel: one private histogram per task
    rows, cols = shape
    x_min, y_min = ext[0], ext[2]
    sx, sy = cols / (ext[1] - ext[0]), rows / (ext[3] - ext[2])
    hists = np.zeros((tasks, rows, cols), dtype=np.uint32)
    for t in prange(tasks):
        hist = hists[t]
        k = t * LANES                                # this task replays segments k .. k+3 in lock-step
        x0, y0, px0, py0 = cps[k, 0], cps[k, 1], cps[k, 2], cps[k, 3]
        x1, y1, px1, py1 = cps[k + 1, 0], cps[k + 1, 1], cps[k + 1, 2], cps[k + 1, 3]
        x2, y2, px2, py2 = cps[k + 2, 0], cps[k + 2, 1], cps[k + 2, 2], cps[k + 2, 3]
        x3, y3, px3, py3 = cps[k + 3, 0], cps[k + 3, 1], cps[k + 3, 2], cps[k + 3, 3]
        fx0, fy0 = force(x0, y0)                     # pure function of (x, y): same bits as in pass 1
        fx1, fy1 = force(x1, y1)
        fx2, fy2 = force(x2, y2)
        fx3, fy3 = force(x3, y3)
        for _ in range(seg):
            hit(x0, y0, x_min, y_min, sx, sy, hist)
            hit(x1, y1, x_min, y_min, sx, sy, hist)
            hit(x2, y2, x_min, y_min, sx, sy, hist)
            hit(x3, y3, x_min, y_min, sx, sy, hist)
            x0, y0, px0, py0, fx0, fy0 = step(x0, y0, px0, py0, fx0, fy0, h)
            x1, y1, px1, py1, fx1, fy1 = step(x1, y1, px1, py1, fx1, fy1, h)
            x2, y2, px2, py2, fx2, fy2 = step(x2, y2, px2, py2, fx2, fy2, h)
            x3, y3, px3, py3, fx3, fy3 = step(x3, y3, px3, py3, fx3, fy3, h)
        if t == tasks - 1:                           # the leftover points belong to the very last segment
            for _ in range(rem):
                hit(x3, y3, x_min, y_min, sx, sy, hist)
                x3, y3, px3, py3, fx3, fy3 = step(x3, y3, px3, py3, fx3, fy3, h)
    return hists


def density(n, tasks):
    segments = tasks * LANES
    seg, rem = divmod(n, segments)
    t0 = time.perf_counter()
    ext, cps = first_pass(*START, DT, segments, seg, rem)
    t1 = time.perf_counter()
    img = second_pass(DT, tasks, seg, rem, cps, ext, SHAPE).sum(axis=0, dtype=np.uint64)
    return img, ext, t1 - t0, time.perf_counter() - t1


@njit
def serial_reference(x, y, px, py, h, n, shape):     # plain two-pass loop: no segments, no lanes. Only for the check.
    rows, cols = shape
    start = (x, y, px, py)
    fx, fy = force(x, y)
    min_x = max_x = x
    min_y = max_y = y
    for _ in range(n):
        min_x, max_x = min(min_x, x), max(max_x, x)
        min_y, max_y = min(min_y, y), max(max_y, y)
        x, y, px, py, fx, fy = step(x, y, px, py, fx, fy, h)
    sx, sy = cols / (max_x - min_x), rows / (max_y - min_y)
    img = np.zeros((rows, cols), dtype=np.uint64)
    x, y, px, py = start
    fx, fy = force(x, y)
    for _ in range(n):
        hit(x, y, min_x, min_y, sx, sy, img)
        x, y, px, py, fx, fy = step(x, y, px, py, fx, fy, h)
    return img, (min_x, max_x, min_y, max_y)


def plot(img, ext, n):                                # hits per pixel, sqrt colour scale (lifts the sparse regions)
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='gainsboro')
    im = ax.imshow(img, origin='lower', extent=ext, cmap='hot', interpolation='none',
                   norm=PowerNorm(gamma=0.5, vmin=0, vmax=int(img.max())))
    ax.set_title(f'Henon-Heiles orbit, leapfrog, E = 1/8, n = {n:_}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(im, ax=ax, location='bottom', label='hits per pixel (sqrt colour scale)')
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    n = 100_000_003                                  # deliberately not divisible by the number of segments
    tasks = 2 * get_num_threads()
    density(1000, 2)                                 # compile once, so the timings below are fair
    serial_reference(*START, DT, 1000, SHAPE)
    t0 = time.perf_counter()
    ref, ref_ext = serial_reference(*START, DT, n, SHAPE)
    print(f'plain serial loop (no segments, no lanes):       total {time.perf_counter() - t0:.2f} s')
    img, ext, t1, t2 = density(n, tasks)
    print(f'{tasks} tasks x {LANES} lanes, {get_num_threads()} threads: pass 1 {t1:.2f} s, pass 2 {t2:.2f} s, total {t1 + t2:.2f} s')
    print('identical to the serial loop (every pixel and every extent):', np.array_equal(img, ref) and ext == ref_ext)
    plot(img, ext, n)
    plt.show()
