# Hopalong Orbit Visualizer & Density Estimator (Python)

> **Despite its common name, Hopalong is not an attractor.**
> The map preserves area (Jacobian determinant 1), so nothing contracts onto an attracting set; the familiar picture is the footprint of a single orbit from (0, 0), and other starting points can give different pictures.

---

## What is the Hopalong Map?

The Hopalong sequence is a classic mathematical formula that generates intricate, organic patterns when plotted over millions of steps. It is defined by the following recursive system:

$$
\large
\begin{cases}
x_{n+1} = y_n - \text{sgn}(x_n) \sqrt{\lvert b x_n - c \rvert} \\
y_{n+1} = a - x_n
\end{cases}
\large
$$

Where:

* The sequence starts from initial coordinates $(x_0, y_0) = (0, 0)$.
* $a$, $b$, and $c$ are parameters that shape the pattern's geometry.
* $\text{sgn}$ is implemented using Python's `math.copysign()` to correctly respect IEEE 754 signed zeros.

---

## The Challenge: Why Parallelizing Chaos is Hard

If you want to render a 100-million-point fractal quickly, you want to split the work across all available CPU cores. However, the Hopalong formula has a strict **domino effect** (a loop-carried dependency): you cannot calculate step #50,000 without knowing the exact output of step #49,999.

Because every point depends strictly on the one before it, standard parallel processing fails—a CPU core cannot simply guess where to start in the middle of a chaotic trajectory.

---

## The Solution: The Checkpoint 2-Pass Approach

To solve this bottleneck without consuming gigabytes of memory, both high-performance scripts (`hopalong_fast.py` and `hopalong_fast_histogram.py`) utilize a **two-pass checkpoint architecture**:

### Phase 1: Pass 1 (The Pathfinder)

* **What it does:** The computer walks through the math sequence *once* from start to finish in a single, lightweight serial loop.
* **What it collects:** It tracks the exact global spatial boundaries (`min_x`, `max_x`, `min_y`, `max_y`) to scale the image, while dropping **"checkpoints"** (breadcrumbs) at regular intervals.
* **What a checkpoint is:** A checkpoint is a tiny record of the exact $(x, y)$ coordinates at the start of a chunk (e.g., step 1,000,000).

### Phase 2: Pass 2 (The Multi-Core Team)

* **What it does:** Multiple CPU cores run simultaneously using Numba's `prange`. Each core grabs its assigned checkpoint, instantly jumps to its starting coordinates, and processes its chunk independently into a private grid.
* **Trajectory Memory Efficiency:** Instead of storing 100 million trajectory points in RAM, the script stores only a small table of checkpoint coordinates. However, pass 2 allocates private pixel histograms proportional to the number of tasks and image dimensions; the histogram version also creates float64 reduction and density arrays. Both scripts report peak process RAM so this working memory is visible.

---

## Comparing the Two Implementations

This project offers two distinct implementations leveraging the same high-speed checkpoint backbone:

| Feature / Metric | Pixel-Based Density (`hopalong_fast.py`) | True Density Histogram (`hopalong_fast_hist.py`) |
| :--- | :--- | :--- |
| **Output Values** | Discrete integer hit counts (N₀) per pixel | Continuous float values (R⁺) representing normalized probability density |
| **Mathematical Formula** | `Hit Count = Σ Hits` | `Density = Raw Hits / (N * Δx * Δy)` |
| **Primary Use Case** | Fast visual exploration, artistic rendering, and qualitative analysis | Scientific data analysis, quantitative comparisons, information entropy, and probability modeling |
| **Execution Speed** | Ultra-fast (multi-core parallel) | Virtually identical (adds a single post-processing normalization step) |
| **Visual Appearance** | Identical to human eye | Identical to human eye |
| **PNG and RAM reporting** | Saves a PNG and reports peak process RAM | Saves a separate `_histogram.png` and reports peak process RAM |

### Key Differences & Practical Guidance

1. **Pixel-Based Density Approximation:** Each cell in the output matrix stores the raw count of times the trajectory stepped on that pixel. It is ideal for visual rendering and heatmap generation where exact probabilities are not required.
2. **True Probability Density Histogram:** Normalizes the raw counts by dividing by the total number of points ($N$) and the spatial bin area ($\Delta x \cdot \Delta y$). The sum of all bins multiplied by their area equals $1.0$, turning the matrix into a true mathematical **Probability Density Function (PDF)**.

---

## Color Mapping Scales (`linear`, `sqrt`, `log`)

When executing either script, you will be prompted to select a color scale mapping. Because fractal point distributions vary wildly—containing both sparse background regions and hyper-dense core lines—selecting the right normalization scale drastically changes how details are revealed:

### 1. `linear`

* **Formula:** $f(d) = \frac{d}{d_{\text{max}}}$
* **Behavior:** Maps pixel/density values directly to colormap brightness in a straight line.
* **Best used for:** Identifying the absolute highest-density core points or "hotspots" of the fractal. Fine background details will often appear dark or invisible.

### 2. `sqrt` (Square Root)

* **Formula:** $f(d) = \sqrt{\frac{d}{d_{\text{max}}}}$ (Gamma = 0.5)
* **Behavior:** Compresses the high-density range and boosts mid-tier densities.
* **Best used for:** Balanced visual exploration. It reveals delicate outer tendrils and faint structures without completely washing out the dense core.

### 3. `log` (Logarithmic)

* **Formula:** $f(d) = \frac{\ln(1 + d)}{\ln(1 + d_{\text{max}})}$
* **Behavior:** Heavily compresses extreme density spikes while dramatically boosting extremely sparse regions (pixels hit only once or twice).
* **Best used for:** Exposing subtle background webs, outer boundary orbits, and hidden low-density structures that are otherwise invisible in linear mode.

---

## Requirements & Execution

### Prerequisites

Install the required dependencies via pip:

```bash
pip install numpy matplotlib numba

```

### Running the Scripts

Both scripts run interactively in your terminal and prompt for parameters ($a, b, c$, total points $n$, and color scale):

```bash
# Run the raw pixel-based density version
python hopalong_fast.py

# Run the true probability density histogram version
python hopalong_fast_hist.py
