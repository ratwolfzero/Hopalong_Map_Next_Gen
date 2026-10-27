import os
import sys
import time
import resource
from math import copysign, sqrt, fabs

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange, get_num_threads

# ----------------------------------------------------------------------------------------------
# Tuning constants (same idea as in the Rust version)
# ----------------------------------------------------------------------------------------------

# Independent orbit segments advanced in lock-step inside one task. The map is a serial
# dependency chain (sqrt latency), so interleaving several segments lets the CPU overlap them.
# NOTE: the kernel below is written out for exactly 4 lanes (explicit scalar variables).
LANES = 4

# Below this many points the parallel machinery isn't worth its set-up cost.
PARALLEL_THRESHOLD = 1_000_000

# Upper bound on the points plotted by one task, so its uint32 pixel counters cannot overflow.
# The per-task histograms are summed into uint64 afterwards, so n itself is not limited.
MAX_POINTS_PER_TASK = 1 << 31

# Where the PNG files are written (same folder as the Rust version).
OUTPUT_DIR = '/Users/ralf/Projects/hopalong_pictures'

# How a pixel's hit count is mapped onto the colour map.
SCALES = ('linear', 'sqrt', 'log')


# ----------------------------------------------------------------------------------------------
# User input (unchanged)
# ----------------------------------------------------------------------------------------------

def validate_input(prompt, input_type=float, check_positive_non_zero=False, min_value=None):
    # Prompt for and return user input validated by type and specific checks.
    while True:
        try:
            value = float(input(prompt))
            if input_type == int:
                if not value.is_integer():
                    raise ValueError('Please enter an integer.')
                value = int(value)
            if check_positive_non_zero and value <= 0:
                raise ValueError('The value must be positive and non-zero.')
            if min_value is not None and value < min_value:
                raise ValueError(f'The value must be at least {min_value}.')
            return value
        except ValueError as e:
            print(f'Invalid input. Please enter a valid {input_type.__name__} value. ({e})')


def validate_attractor_parameters(a, b, c):
    while a == 0 and c == 0:
        print('Invalid combination of parameters: a=0, b=0, c=0 or a=0, b=any, c=0')
        c = validate_input('Enter a float value for "c": ')
    return a, b, c


def validate_scale(prompt, default='linear'):
    while True:
        value = input(prompt).strip().lower() or default
        if value in SCALES:
            return value
        print(f'Invalid input. Please enter one of: {", ".join(SCALES)}.')


def get_attractor_parameters():
    a = validate_input('Enter a float value for "a": ')
    b = validate_input('Enter a float value for "b": ')
    c = validate_input('Enter a float value for "c": ')
    a, b, c = validate_attractor_parameters(a, b, c)
    n = validate_input('Enter a positive integer value > 1000 for "n": ', int, True, 1000)
    scale = validate_scale('Colour scale (linear / sqrt / log) [linear]: ')
    return {'a': a, 'b': b, 'c': c, 'n': n, 'scale': scale}


# ----------------------------------------------------------------------------------------------
# Orbit plan: the n points are cut into `segments` consecutive pieces of `seg` points each
# (the last piece also takes the `rem` leftover points). Pass 1 walks the orbit once, serially,
# and records the bounds plus the state at the start of every piece. Pass 2 can then replay the
# pieces independently - and in parallel - and visits exactly the same points as one sequential
# run would.
# ----------------------------------------------------------------------------------------------

def make_plan(n):
    if n >= PARALLEL_THRESHOLD:
        tasks = max(2 * get_num_threads(), -(-n // MAX_POINTS_PER_TASK))  # 2x threads, ceil-div
    else:
        tasks = 1
    segments = tasks * LANES
    return tasks, segments, n // segments, n % segments


@njit(inline='always')
def _step(x, y, a, b, c):
    # signum respecting IEEE 754 (signed zero), same as the original
    return y - copysign(1.0, x) * sqrt(fabs(b * x - c)), a - x


@njit(cache=True)
def first_pass(a, b, c, segments, seg, rem):
    # Pass 1: extents of the orbit plus the start state of every segment (a tiny array).
    checkpoints = np.empty((segments, 2))
    x = 0.0
    y = 0.0
    min_x = max_x = 0.0  # (0, 0) is the first orbit point, so it is a valid starting value
    min_y = max_y = 0.0

    for k in range(segments):
        checkpoints[k, 0] = x
        checkpoints[k, 1] = y
        count = seg + rem if k == segments - 1 else seg
        for _ in range(count):
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
            x, y = _step(x, y, a, b, c)

    return (min_x, max_x, min_y, max_y), checkpoints


@njit(inline='always')
def _hit(x, y, min_x, min_y, scale_x, scale_y, hist):
    h, w = hist.shape
    # np.rint rounds half-to-even like Python's round(). Range-check in float *before* the
    # int cast, so NaN/inf points are discarded instead of hitting a garbage index.
    px = np.rint((x - min_x) * scale_x)
    py = np.rint((y - min_y) * scale_y)
    if 0.0 <= px < w and 0.0 <= py < h:
        hist[int(py), int(px)] += 1


@njit(cache=True, parallel=True)
def second_pass(a, b, c, tasks, seg, rem, checkpoints, extents, image_size):
    # Pass 2: every task replays LANES consecutive segments in lock-step into its OWN uint32
    # histogram (no sharing -> no locks, no atomics, cache friendly).
    h, w = image_size
    min_x, max_x, min_y, max_y = extents
    scale_x = (w - 1) / (max_x - min_x)  # column
    scale_y = (h - 1) / (max_y - min_y)  # row

    hists = np.zeros((tasks, h, w), dtype=np.uint32)

    for t in prange(tasks):
        hist = hists[t]
        k = t * LANES
        x0, y0 = checkpoints[k, 0], checkpoints[k, 1]
        x1, y1 = checkpoints[k + 1, 0], checkpoints[k + 1, 1]
        x2, y2 = checkpoints[k + 2, 0], checkpoints[k + 2, 1]
        x3, y3 = checkpoints[k + 3, 0], checkpoints[k + 3, 1]

        for _ in range(seg):
            # four independent dependency chains -> the CPU can overlap their sqrt latencies
            _hit(x0, y0, min_x, min_y, scale_x, scale_y, hist)
            _hit(x1, y1, min_x, min_y, scale_x, scale_y, hist)
            _hit(x2, y2, min_x, min_y, scale_x, scale_y, hist)
            _hit(x3, y3, min_x, min_y, scale_x, scale_y, hist)
            x0, y0 = _step(x0, y0, a, b, c)
            x1, y1 = _step(x1, y1, a, b, c)
            x2, y2 = _step(x2, y2, a, b, c)
            x3, y3 = _step(x3, y3, a, b, c)

        if t == tasks - 1:  # leftover points belong to the very last segment of the orbit
            for _ in range(rem):
                _hit(x3, y3, min_x, min_y, scale_x, scale_y, hist)
                x3, y3 = _step(x3, y3, a, b, c)

    return hists


def compute_trajectory_image(a, b, c, n, image_size):
    # Returns (image as uint64, extents, (pass1_seconds, pass2_seconds)).
    tasks, segments, seg, rem = make_plan(n)

    t0 = time.perf_counter()
    extents, checkpoints = first_pass(a, b, c, segments, seg, rem)
    t1 = time.perf_counter()
    hists = second_pass(a, b, c, tasks, seg, rem, checkpoints, extents, image_size)
    image = hists.sum(axis=0, dtype=np.uint64)
    t2 = time.perf_counter()
    return image, extents, (t1 - t0, t2 - t1)


def warm_up():
    # Trigger JIT compilation (or loading from the on-disk cache) before timing anything.
    compute_trajectory_image(1.0, 1.0, 1.0, 1000, (2, 2))


# ----------------------------------------------------------------------------------------------
# Plotting (unchanged apart from the cosmetic bits)
# ----------------------------------------------------------------------------------------------

def setup_plot(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def make_norm(image, scale):
    # Same three mappings as the Rust version (relative brightness 0..1):
    #   linear: d/max      sqrt: sqrt(d/max)      log: ln(1+d)/ln(1+max)
    max_hit = max(int(np.max(image)), 1)
    if scale == 'sqrt':
        return mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=max_hit)
    if scale == 'log':
        return mcolors.FuncNorm((np.log1p, np.expm1), vmin=0, vmax=max_hit)
    return mcolors.Normalize(vmin=0, vmax=max_hit)


def colorbar(fig, ax, img, norm, scale):
    cbar = fig.colorbar(img, ax=ax, location='bottom')
    cbar.set_label(f'Pixel Density (scale: {scale}, 0 - max)')
    # Ticks evenly spaced on the *colour bar* (not in hit counts), labelled with the hit count.
    ticks = norm.inverse(np.linspace(0.0, 1.0, 8))
    labels = np.rint(ticks).astype(np.int64)
    _, keep = np.unique(labels, return_index=True)  # drop duplicate labels (tiny max counts)
    keep = np.sort(keep)
    cbar.set_ticks(ticks[keep])
    cbar.set_ticklabels([f'{int(v):,}' for v in labels[keep]])


def render_trajectory_image(image, extents, params, color_map, norm):
    fig = plt.figure(figsize=(8, 8), facecolor='gainsboro')
    ax = fig.add_subplot(111)
    img = ax.imshow(image, origin='lower', cmap=color_map, norm=norm, extent=extents,
                    interpolation='none')

    title = (f'Hopalong Attractor@ratwolf@2024\n'
             f'Params: a={params["a"]}, b={params["b"]}, c={params["c"]}, n={params["n"]:_}')
    setup_plot(ax, title, 'X (Cartesian)', 'Y (Cartesian)')
    colorbar(fig, ax, img, norm, params['scale'])

    plt.tight_layout()
    plt.show()


def save_png(image, params, color_map, norm, directory=OUTPUT_DIR):
    # Raw picture only (no axes / colour bar), one image pixel per grid cell, like the Rust version.
    suffix = '' if params['scale'] == 'linear' else f'_{params["scale"]}'
    name = (f'py_num={params["n"]}_a={params["a"]}_b={params["b"]}_c={params["c"]}{suffix}.png')
    try:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, name)
        rgba = plt.get_cmap(color_map)(norm(image))
        plt.imsave(path, rgba, origin='lower')  # row 0 = bottom, as in imshow(origin='lower')
        print(f'Image saved to: {path}')
        return path
    except OSError as e:
        print(f'Could not save image to {directory}: {e}')
        return None


def display_resource_usage(wall_seconds, pass_times):
    # ru_maxrss is in bytes on macOS but in kilobytes on Linux
    divisor = 1024.0 * 1024.0 if sys.platform == 'darwin' else 1024.0
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor

    print(f'Pass 1 (extents + checkpoints, serial): {pass_times[0]:.2f} s')
    print(f'Pass 2 (parallel plot):                 {pass_times[1]:.2f} s')
    print(f'Wall-clock time (compute): {wall_seconds:.2f} s')
    print(f'Memory (RAM): {mem_mb:.2f} MByte used (peak)')


def main(image_size=(1000, 1000), color_map='hot'):
    try:
        params = get_attractor_parameters()
        warm_up()

        # Wall-clock time: process_time() would add up the CPU time of ALL threads.
        start = time.perf_counter()
        image, extents, pass_times = compute_trajectory_image(
            params['a'], params['b'], params['c'], params['n'], image_size)
        wall = time.perf_counter() - start

        display_resource_usage(wall, pass_times)

        norm = make_norm(image, params['scale'])
        save_png(image, params, color_map, norm)
        render_trajectory_image(image, extents, params, color_map, norm)

    except Exception as e:
        print(f'An error occurred: {e}')


if __name__ == '__main__':
    main()
