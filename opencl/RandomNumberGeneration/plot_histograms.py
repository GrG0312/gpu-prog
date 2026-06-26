"""
plot_histograms.py
------------------
Reads the CSV histogram files produced by the C++ program and visualizes
the distribution of each RNG's output as a bar chart.

Each CSV contains two columns:
    bin: index (0 to HIST_BINS-1)
    count: number of generated values in that bin

For a uniform RNG all bins should have approximately equal counts.
The dashed horizontal line marks the expected frequency if the
distribution were perfectly uniform.

Usage:
    python plot_histograms.py

Output:
    histograms.png  saved to the current working directory
"""

import os
import sys
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------
#  CSV files to plot, in display order.
#  key   = subplot title
#  value = filename produced by the C++ program
# ---------------------------------------------------------------
CSV_FILES = {
    "GPU LCG":              "histogram_lcg.csv",
    "GPU XORShift":         "histogram_xorshift.csv",
    "GPU Mersenne Twister": "histogram_mt.csv",
    "CPU std::mt19937":     "histogram_cpu_mt19937.csv",
}

COLORS = ["steelblue", "tomato", "seagreen", "darkorange"]


# ---------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------
def load_csv(path):
    """Returns (bins, counts) as two lists of ints."""
    bins, counts = [], []

    with open(path, newline = "") as f:
        for row in csv.DictReader(f):
            bins.append(int(row["bin"]))
            counts.append(int(row["count"]))

    return bins, counts


def check_files():
    """Exits with a clear message if any CSV is missing."""
    missing = [p for p in CSV_FILES.values() if not os.path.exists(p)]
    if missing:
        print("The following CSV files were not found.")
        print("Run the C++ program first to generate them:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)


# ---------------------------------------------------------------
#  Plot
# ---------------------------------------------------------------
def main():
    check_files()

    n_plots = len(CSV_FILES)
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols  # ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    fig.suptitle(
        "RNG Output Distribution — observed frequency per bin\n"
        "(dashed line = expected frequency for a perfectly uniform distribution)",
        fontsize = 12,
        fontweight = "bold"
    )

    for ax, (title, path), color in zip(axes.flat, CSV_FILES.items(), COLORS):
        bins, counts = load_csv(path)

        total = sum(counts)
        n_bins = len(bins)
        expected = total / n_bins

        # Bar chart of observed frequencies
        ax.bar(bins, counts, color = color, alpha = 0.75, label = "Observed")

        # Expected frequency reference line
        ax.axhline(
            expected,
            color = "black",
            linewidth = 1.2,
            linestyle = "--",
            label = f"Expected  ({expected:,.0f})"
        )

        ax.set_title(title, fontsize = 11, fontweight = "bold")
        ax.set_xlabel("Bin index")
        ax.set_ylabel("Count")

        # Format y-axis with thousands separator
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
        )

        ax.legend(fontsize = 9)

        # Annotate with N and deviation info
        min_c   = min(counts)
        max_c   = max(counts)
        dev_pct = (max_c - min_c) / expected * 100.0
        info    = f"N = {total:,}\nmin = {min_c:,}  max = {max_c:,}\nmax deviation = {dev_pct:.1f}%"
        ax.text(
            0.98, 0.97, info,
            transform = ax.transAxes,
            ha = "right", va = "top",
            fontsize = 8,
            family = "monospace",
            bbox = dict(boxstyle = "round,pad=0.4", fc = "white", alpha = 0.8)
        )

    # Hide any unused subplots if n_plots is odd
    for ax in axes.flat[n_plots:]:
        ax.set_visible(False)

    plt.tight_layout()

    out = "histograms.png"
    plt.savefig(out, dpi = 150)
    print(f"Saved -> {out}")
    plt.show()


if __name__ == "__main__":
    main()