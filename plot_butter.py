import json
import matplotlib.pyplot as plt
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

plt.figure(figsize=(19.2, 10.8), layout='constrained')
plt.plot(frames, scores, linewidth=1.0)
plt.xlim(0, len(frames) - 1 if frames else 0)
plt.ylim(0, 5)
plt.xlabel('Frame', fontdict={'fontsize': 18, 'color': 'darkblue'})
plt.ylabel('Butteraugli 3-norm Score', fontdict={'fontsize': 22, 'color': 'darkblue'})
plt.title('Butteraugli Scores per Frame', fontdict={'fontsize': 22, 'color': 'darkblue'})
plt.grid(True)

plt.savefig(png_file)
plt.close()

print(f"Plot successfully saved to '{png_file}'")
