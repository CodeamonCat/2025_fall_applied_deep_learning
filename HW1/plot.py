import json
import matplotlib.pyplot as plt
import pandas as pd
import os

# Path to your trainer's output directory
output_dir = "./result_qa_hfl_chinese-lert-base"  # replace with your training_args.output_dir

# Load the training log (trainer logs metrics in 'trainer_state.json')
trainer_state_file = os.path.join(output_dir, "trainer_state.json")
if not os.path.exists(trainer_state_file):
    raise FileNotFoundError(f"{trainer_state_file} not found!")

with open(trainer_state_file, "r", encoding="utf-8") as f:
    trainer_state = json.load(f)

# Extract logged metrics
log_history = trainer_state.get("log_history", [])

# Convert to DataFrame for easier plotting
df = pd.DataFrame(log_history)

# Filter only relevant columns (step, loss, exact_match)
if "loss" not in df.columns:
    print("Warning: 'loss' not found in log history.")
if "eval_exact_match" not in df.columns:
    print("Warning: 'eval_exact_match' not found in log history.")

# Plot training loss
if "loss" in df.columns:
    plt.figure(figsize=(8, 5))
    plt.plot(df["step"], df["loss"], marker='o')
    plt.title("Training Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "training_loss_curve.png"))
    plt.show()

# Plot validation exact match
if "eval_exact_match" in df.columns:
    plt.figure(figsize=(8, 5))
    # Filter only eval steps
    eval_df = df.dropna(subset=["eval_exact_match"])
    plt.plot(eval_df["step"], eval_df["eval_exact_match"], marker='o', color="orange")
    plt.title("Validation Exact Match Curve")
    plt.xlabel("Step")
    plt.ylabel("Exact Match")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "validation_exact_match_curve.png"))
    plt.show()
