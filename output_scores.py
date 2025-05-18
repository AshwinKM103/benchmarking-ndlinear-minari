import os
import re
from pathlib import Path

# ==== USER INPUT ====
input_dir = "/data1/home/nitinvetcha/Ashwin_KM_Code/benchmarking-ndlinear-minari/wandb_logs/AWAC-Minari"        # e.g., "./results"
output_dir = "/data1/home/nitinvetcha/Ashwin_KM_Code/benchmarking-ndlinear-minari/output_scores_final"       # e.g., "./top_means"
output_filename = "top_awac_scores.txt"          # e.g., "top_bc_scores.txt"
# =====================

# Ensure output directory exists
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Compile regex for mean and std extraction
pattern = re.compile(r"Evaluation over 10 episodes:\s*mean:\s*([-+]?[0-9]*\.?[0-9]+),\s*std:\s*([-+]?[0-9]*\.?[0-9]+)")


output_lines = []

# Process files starting with "BC"
for fname in os.listdir(input_dir):
    if fname.startswith('AWAC'):  # assuming .txt files
        full_path = os.path.join(input_dir, fname)
        with open(full_path, "r") as f:
            content = f.read()
            matches = pattern.findall(content)
            means_stds = [(float(m), float(s)) for m, s in matches]

            if means_stds:
                top5 = sorted(means_stds, key=lambda x: x[0], reverse=True)[:5]
                output_lines.append(f"{fname}")
                for mean, std in top5:
                    output_lines.append(f"<td>{mean} &plusmn; {std}</td>")
                output_lines.append("")  # Blank line between files

# Write to output file
with open(os.path.join(output_dir, output_filename), "w") as out_f:
    out_f.write("\n".join(output_lines))

print(f"Top 5 scores per file written to {os.path.join(output_dir, output_filename)}")
