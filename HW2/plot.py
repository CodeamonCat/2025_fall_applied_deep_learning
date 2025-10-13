import argparse
import json
import matplotlib.pyplot as plt
import os
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="./result",
        required=False,
        help="Input folder name (default: ./result)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./result",
        required=False,
        help="Output folder name (default: ./result)",
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()

    trainer_state_file = args.input + "/trainer_state.json"
    if not os.path.exists(trainer_state_file):
        raise FileNotFoundError(f"{trainer_state_file} not found!")

    with open(trainer_state_file, "r") as f:
        trainer_state = json.load(f)

    steps = []
    losses = []
    eval_perplexities = []

    for entry in trainer_state["log_history"]:
        if "loss" in entry:
            steps.append(entry["step"])
            losses.append(entry["loss"])
        if "eval_perplexity" in entry:
            eval_perplexities.append((entry["step"], entry["eval_perplexity"]))

    # Plot training loss
    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, label="Training Loss")
    if eval_perplexities:
        eval_steps, eval_ppls = zip(*eval_perplexities)
        plt.plot(eval_steps, eval_ppls, label="Evaluation Perplexity")
    plt.xlabel("Steps")
    plt.ylabel("Loss / Perplexity")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(args.output + "/learning_curve.png")


if __name__ == "__main__":
    main()
