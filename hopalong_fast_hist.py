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
# Tuning constants
# ----------------------------------------------------------------------------------------------
LANES = 4
PARALLEL_THRESHOLD = 1_000_000
MAX_POINTS_PER_TASK = 1 << 31

# How a pixel's density is mapped onto the color map
SCALES = ('linear', 'sqrt', 'log')


# ----------------------------------------------------------------------------------------------
# User input validation
# ----------------------------------------------------------------------------------------------
def validate_input(prompt, input_type=float, check_positive_non_zero=False, min_value=None):
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
    scale = validate_scale('Color scale (linear / sqrt / log) [linear]: ')
    return {'a': a, 'b': b, 'c': c, 'n': n, 'scale': scale}


# ----------------------------------------------------------------------------------------------
# Pass 1: Plan and Extents Checkpoint Discovery
# ----------------------------------------------------------------------------------------------
def make_plan(n):
    if n >= PARALLEL_THRESHOLD:
        tasks = max(2 * get_num_threads(), -(-n // MAX_POINTS_PER_TASK))
    else:
        tasks = 1
    segments = tasks * LANES
    return tasks, segments, n // segments, n % segments


@njit(inline='always')
def _step(x, y, a, b, c):
    return y - copysign(1.0, x) * sqrt(fabs(b * x - c)), a - x


@njit(cache=True)
def first_pass(a, b, c, segments, seg, rem):
    checkpoints = np.empty((segments, 2))
    x = 0.0
    y = 0.0
    min_x = max_x = 0.0
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


# ----------------------------------------------------------------------------------------------
# Pass 2: Parallel Binning into Private Histograms
# ----------------------------------------------------------------------------------------------
@njit(inline='always')
def _hit(x, y, min_x, min_y, scale_x, scale_y, hist):
    h, w = hist.shape
    px = np.rint((x - min_x) * scale_x)
    py = np.rint((y - min_y) * scale_y)
    if 0.0 <= px < w and 0.0 <= py < h:
        hist[int(py), int(px)] += 1


@njit(cache=True, parallel=True)
def second_pass(a, b, c, tasks, seg, rem, checkpoints, extents, image_size):
    h, w = image_size
    min_x, max_x, min_y, max_y = extents
    scale_x = (w - 1) / (max_x - min_x)
    scale_y = (h - 1) / (max_y - min_y)

    hists = np.zeros((tasks, h, w), dtype=np.uint32)

    for t in prange(tasks):
        hist = hists[t]
        k = t * LANES
        x0, y0 = checkpoints[k, 0], checkpoints[k, 1]
        x1, y1 = checkpoints[k + 1, 0], checkpoints[k + 1, 1]
        x2, y2 = checkpoints[k + 2, 0], checkpoints[k + 2, 1]
        x3, y3 = checkpoints[k + 3, 0], checkpoints[k + 3, 1]

        for _ in range(seg):
            _hit(x0, y0, min_x, min_y, scale_x, scale_y, hist)
            _hit(x1, y1, min_x, min_y, scale_x, scale_y, hist)
            _hit(x2, y2, min_x, min_y, scale_x, scale_y, hist)
            _hit(x3, y3, min_x, min_y, scale_x, scale_y, hist)
            x0, y0 = _step(x0, y0, a, b, c)
            x1, y1 = _step(x1, y1, a, b, c)
            x2, y2 = _step(x2, y2, a, b, c)
            x3, y3 = _step(x3, y3, a, b, c)

        if t == tasks - 1:
            for _ in range(rem):
                _hit(x3, y3, min_x, min_y, scale_x, scale_y, hist)
                x3, y3 = _step(x3, y3, a, b, c)

    return hists


# ----------------------------------------------------------------------------------------------
# Core Execution & True Density Normalization
# ----------------------------------------------------------------------------------------------
def compute_true_histogram(a, b, c, n, image_size):
    tasks, segments, seg, rem = make_plan(n)

    t0 = time.perf_counter()
    extents, checkpoints = first_pass(a, b, c, segments, seg, rem)
    t1 = time.perf_counter()

    hists = second_pass(a, b, c, tasks, seg, rem, checkpoints, extents, image_size)
    raw_counts = hists.sum(axis=0, dtype=np.float64)
    t2 = time.perf_counter()

    # Normalization step to create a True Probability Density Histogram:
    # Density = Raw Count / (Total Points (n) * Bin Area)
    min_x, max_x, min_y, max_y = extents
    h, w = image_size
    dx = (max_x - min_x) / (w - 1)
    dy = (max_y - min_y) / (h - 1)
    bin_area = dx * dy

    density_histogram = raw_counts / (n * bin_area)

    return density_histogram, extents, (t1 - t0, t2 - t1)


def warm_up():
    compute_true_histogram(1.0, 1.0, 1.0, 1000, (2, 2))


# ----------------------------------------------------------------------------------------------
# Plotting Utilities (Supporting linear, sqrt, and log scales)
# ----------------------------------------------------------------------------------------------
def make_norm(density_hist, scale):
    max_density = max(float(np.max(density_hist)), 1e-12)
    if scale == 'sqrt':
        return mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=max_density)
    if scale == 'log':
        return mcolors.FuncNorm((np.log1p, np.expm1), vmin=0, vmax=max_density)
    return mcolors.Normalize(vmin=0, vmax=max_density)


def colorbar(fig, ax, img, norm, scale):
    cbar = fig.colorbar(img, ax=ax, location='bottom')
    cbar.set_label(f'Probability Density (scale: {scale}, 0 - max)')
    
    # Generate 8 evenly spaced ticks along the colorbar axis
    ticks = norm.inverse(np.linspace(0.0, 1.0, 8))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{v:.2e}' for v in ticks])


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def main(image_size=(1000, 1000), color_map='hot'):
    try:
        params = get_attractor_parameters()
        warm_up()

        start = time.perf_counter()
        density_hist, extents, pass_times = compute_true_histogram(
            params['a'], params['b'], params['c'], params['n'], image_size)
        wall = time.perf_counter() - start

        print(f'Pass 1 (extents + checkpoints, serial): {pass_times[0]:.2f} s')
        print(f'Pass 2 (parallel histogram binning):    {pass_times[1]:.2f} s')
        print(f'Total Wall-clock time:                  {wall:.2f} s')

        # Create normalization rule based on selected scale
        norm = make_norm(density_hist, params['scale'])

        fig = plt.figure(figsize=(8, 8), facecolor='gainsboro')
        ax = fig.add_subplot(111)
        img = ax.imshow(density_hist, origin='lower', cmap=color_map, norm=norm,
                        extent=extents, interpolation='none')

        title = (f'Hopalong Probability Density Histogram\n'
                 f'Params: a={params["a"]}, b={params["b"]}, c={params["c"]}, n={params["n"]:_}')
        ax.set_title(title)
        ax.set_xlabel('X (Cartesian)')
        ax.set_ylabel('Y (Cartesian)')

        colorbar(fig, ax, img, norm, params['scale'])

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f'An error occurred: {e}')


if __name__ == '__main__':
    main()
