import wandb
import os
import shutil

# Authenticate with Weights & Biases (if needed)
wandb.login()

# Set your W&B entity, project, and group name
entity = "ashwin123robotix-indian-institute-of-science"
project = "CORL"
group_name = "TD3_BC-Minari"

# Initialize W&B API
api = wandb.Api()

# Get all runs under the specified group
runs = api.runs(f"{entity}/{project}", filters={"group": group_name})

# Create a directory to store logs
group_dir = os.path.join("wandb_logs", group_name)
os.makedirs(group_dir, exist_ok=True)

# Download and save output.log from each run
for run in runs:
    print(f"Processing run: {run.name}")
    for file in run.files():
        if file.name == "output.log":
            try:
                temp_path = os.path.join("/tmp", "output.log")
                dest_path = os.path.join(group_dir, f"{run.name}.log")
                
                print(f"Downloading to: {temp_path}")
                file.download(replace=True, root="/tmp")  # Download to /tmp

                print(f"Moving to: {dest_path}")
                shutil.move(temp_path, dest_path)

            except Exception as e:
                print(f"❌ Failed to process run {run.name}: {e}")
