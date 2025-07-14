import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import sys

if len(sys.argv) != 3:
    print("Usage: python plot_ssimu2.py <input_json_file> <output_png_file>")
    sys.exit(1)

json_file = sys.argv[1]
png_file = sys.argv[2]

try:
    with open(json_file, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: The file '{json_file}' was not found.")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from the file '{json_file}'.")
    sys.exit(1)

try:
    scores = [item[0] for item in data]
except (TypeError, IndexError):
    print("Error: The JSON data is not in the expected format (a list of lists/tuples).")
    sys.exit(1)

frames = range(len(scores))

fig, ax = plt.subplots(figsize=(19.2, 10.8), layout='constrained')
ax.plot(frames, scores, linewidth=1.0)

###     Calculate Statistics
overall_mean = np.mean(scores)
percentile_5 = np.percentile(scores, 5)
percentile_95 = np.percentile(scores, 95)
overall_max = np.max(scores)
overall_min = np.min(scores)

###     Plot Reference Lines
ax.axhline(overall_mean, color='r', linestyle='--', linewidth=1, label='Overall Mean')
ax.axhline(percentile_5, color='purple', linestyle='--', linewidth=1, label='5th Percentile')
ax.axhline(percentile_95, color='purple', linestyle='--', linewidth=1, label='95th Percentile')
ax.axhline(overall_max, color='g', linestyle='--', linewidth=1, label='Overall Max')
ax.axhline(overall_min, color='g', linestyle='--', linewidth=1, label='Overall Min')

###     Calculate Rolling Statistics
scores_series = pd.Series(scores)

###     Span for EWMA and rolling averages
span_size = 100
offset = span_size // 2

###     Vibe Coding begins here
def custom_rolling_quantile(series, window_size, quantile):
    n = len(series)
    offset = window_size // 2

    # Calculate on original series
    result = series.rolling(window=window_size, min_periods=1).quantile(quantile)
    # Extend it
    result = result.reindex(range(n + offset))

    for i in range(n, n + offset):
        # This is for a trailing window at index i
        window_start = i - window_size + 1
        available_data = series.iloc[max(0, window_start):n]

        if available_data.empty:
            result.iloc[i] = result.iloc[n-1]
            continue

        # "median of the frames in the preceeding part of the window"
        # For a trailing window, the whole window is "preceding".
        # So we take the median of all available data in the window.
        median_val = available_data.median()

        num_missing = window_size - len(available_data)
        padding = pd.Series([median_val] * num_missing)
        window_with_padding = pd.concat([available_data, padding])

        result.iloc[i] = window_with_padding.quantile(quantile)

    return result

def custom_ewm(series, span):
    n = len(series)
    offset = span // 2

    # To calculate EWM for extended part, we need to extend the series.
    # Let's use the recursive median padding.
    extended_series = pd.Series(np.nan, index=range(n + offset), dtype=np.float64)
    extended_series.iloc[0:n] = series
    for i in range(n, n + offset):
        # For EWM, the "window" is not well-defined. Let's use span as window size.
        preceding_data = extended_series.iloc[max(0, i - span):i]
        if preceding_data.empty:
            # Fallback for the very first padded points if span is large
            median_val = series.median()
        else:
            median_val = preceding_data.median()
        extended_series.iloc[i] = median_val

    return extended_series.ewm(span=span, adjust=True, min_periods=1).mean()

###     End Vibe Coding

ewm_mean = custom_ewm(scores_series, span_size)
rolling_p10 = custom_rolling_quantile(scores_series, span_size, 0.10)
rolling_p90 = custom_rolling_quantile(scores_series, span_size, 0.90)

x_rolling = np.array(range(len(ewm_mean)))
x_shifted = x_rolling - offset

ax.plot(x_shifted, ewm_mean, label=f'Exponentially Weighted Moving Average, Span={span_size}', color='darkred', linewidth=1)
ax.plot(x_shifted, rolling_p10, label=f'Rolling 10th Percentile, Span={span_size}', color='indigo', linewidth=1)
ax.plot(x_shifted, rolling_p90, label=f'Rolling 90th Percentile, Span={span_size}', color='indigo', linewidth=1)

###     Set Plot Labels and Limits ---
ax.set_xlim(0, len(frames) - 1 if frames else 0)
ax.set_ylim(0, 5)
ax.set_xlabel('Frame', fontdict={'fontsize': 18, 'color': 'darkblue'})
ax.set_ylabel('Butteraugli 3-norm Score', fontdict={'fontsize': 22, 'color': 'darkblue'})
ax.set_title('Butteraugli Scores per Frame', fontdict={'fontsize': 22, 'color': 'darkblue'})
ax.grid(True)
ax.legend()

plt.savefig(png_file)
plt.close()

print(f"Plot successfully saved to '{png_file}'")
