import os
import re

base_dir = "/data1/home/nitinvetcha/Ashwin_KM_Code/benchmarking-ndlinear-minari/wandb_logs/IQL-Minari"

REF_MIN_SCORE = {
    'halfcheetah': -79.20,
    'hopper': 395.64,
    'walker2d': 0.20,
}

REF_MAX_SCORE = {
    'halfcheetah': 16584.93,
    'hopper': 4376.33,
    'walker2d': 6972.80,
}

# Matches: mean: 123.45, std: 67.89
pattern = re.compile(r"Evaluation over 10 episodes:\s*mean:\s*(-?\d+\.?\d*),\s*std:\s*(\d+\.?\d*)")


for filename in os.listdir(base_dir):
    # Only process filenames like BC_0.1-halfcheetah-expert--ndlinear
    if re.match(r"IQL(_Minari)?", filename):
        match_env = re.search(r"-(halfcheetah|hopper|walker2d)-", filename)
        if not match_env:
            continue
        env = match_env.group(1)
        full_path = os.path.join(base_dir, filename)

        with open(full_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            match = pattern.search(line)
            if match:
                mean = float(match.group(1))
                std = float(match.group(2))
                ref_min = REF_MIN_SCORE[env]
                ref_max = REF_MAX_SCORE[env]

                # Normalize mean
                norm_mean = 100 * (mean - ref_min) / (ref_max - ref_min)

                # Normalize std using: 100 × std / (expert - random)
                norm_std = 100 * std / (ref_max - ref_min)

                new_line = f"Evaluation over 10 episodes: mean: {norm_mean:.3f}, std: {norm_std:.3f}\n"
                new_lines.append(new_line)
            else:
                new_lines.append(line)
            
            

        with open(full_path, 'w') as f:
            f.writelines(new_lines)
        
        print(f"Processed and normalized: {filename}")

print("Selective normalization complete.")
