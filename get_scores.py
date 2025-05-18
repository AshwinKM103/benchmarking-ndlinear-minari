import pandas as pd
import numpy as np

# Placeholder file paths
file1_path = "bc_0.1_score.csv"  # This is the file with means
file2_path = "bc_0.1_std.csv"  # This is the file with stds

# Step 1: Read both files
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

max_cols_mean = df1.shape[1]
max_cols_std = df2.shape[1]
print(f"Number of columns in {file1_path}: {max_cols_mean}")
print(f"Number of columns in {file2_path}: {max_cols_std}")

cols_to_keep = [0] + list(range(1, 27, 3))  # Zero-indexed
df_mean = pd.read_csv(file1_path).iloc[:, cols_to_keep]
df_std = pd.read_csv(file2_path).iloc[:, cols_to_keep]
df_mean.columns = df_mean.columns.str.replace(r'\s+', '', regex=True).str.strip()
df_std.columns = df_std.columns.str.replace(r'\s+', '', regex=True).str.strip()
print(df_mean.columns)
print(df_std.columns)

results = []

# Process each column (excluding first, which is index/timestep/etc.)
for i in range(1, len(df_mean.columns)):
    col_name = df_mean.columns[i]
    
    # Top 5 values and their indices
    top5_idx = df_mean[col_name].nlargest(5).index
    top5_means = df_mean.loc[top5_idx, col_name]
    col_std_name = col_name.replace("_score", "_std")
    top5_stds = df_std.loc[top5_idx, col_std_name]
    
    # Compute averages
    avg_mean = top5_means.mean()
    avg_std = top5_stds.mean()
    
    # Append result
    results.append({
        "name": col_name,
        "mean": round(avg_mean, 4),
        "std": round(avg_std, 4)
    })


# Convert to DataFrame and append to CSV
summary_df = pd.DataFrame(results)
summary_file = "final_scores.csv"

try:
    existing_df = pd.read_csv(summary_file)
    updated_df = pd.concat([existing_df, summary_df], ignore_index=True)
    updated_df.to_csv(summary_file, index=False)
except FileNotFoundError:
    summary_df.to_csv(summary_file, index=False)

