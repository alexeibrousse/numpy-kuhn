"""
Analyzes logged data from a training run of the NumNet agent and generates graphs.
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt

COLUMN_ALIASES = {
    "reward_mean": "reward",
    "baseline_mean": "baseline",
    "entropy_mean": "entropy",
}

REQUIRED_COLUMNS = {
    "reward": 0.0,
    "baseline": 0.0,
    "grad_norm": 0.0,
    "entropy": 0.0,
    "learning_rate": 0.0,
    "p_check": 0.0,
    "p_bet": 0.0,
    "p_call": 0.0,
    "p_fold": 0.0,
    "call_rate": 0.0,
    "bluff_rate": 0.0,
    "value_bet_rate": 0.0,
}


def _resolve_paths(run_dir: str) -> tuple[str, str, str]:
    training_dir = os.path.join(run_dir, "training")
    summary_path = os.path.join(training_dir, "training_summary.csv")
    full_data_path = os.path.join(training_dir, "full_training_data.csv")
    return training_dir, summary_path, full_data_path


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Align column names and fill any missing metrics with defaults."""
    df = df.copy()

    for old, new in COLUMN_ALIASES.items():
        if old in df and new not in df:
            df[new] = df[old]

    if "episode" not in df:
        df["episode"] = range(1, len(df) + 1)

    for col, default in REQUIRED_COLUMNS.items():
        if col not in df:
            df[col] = default

    numeric_cols = [c for c in df.columns if c in REQUIRED_COLUMNS or c == "episode"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return df


def load_training_dataframe(run_dir: str, provided_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Load training metrics from disk (or normalize a provided dataframe).
    Prefers the aggregated training_summary.csv produced by the trainer,
    but will fall back to the legacy full_training_data.csv if present.
    """
    if provided_df is not None:
        return _normalize_dataframe(provided_df)

    training_dir, summary_path, full_data_path = _resolve_paths(run_dir)
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
    elif os.path.exists(full_data_path):
        df = pd.read_csv(full_data_path)
    else:
        raise FileNotFoundError(
            f"Could not find training logs in {training_dir}. Expected one of: "
            f"{summary_path} or {full_data_path}"
        )
    return _normalize_dataframe(df)


def analyze(df: pd.DataFrame, run_dir: str) -> None:
    """Computes graphs and summary statistics from the training data."""
    training_dir, _, _ = _resolve_paths(run_dir)
    os.makedirs(training_dir, exist_ok=True)

    if df.empty:
        raise ValueError("Training log is empty; no data to analyze.")

    n_epochs = len(df)
    interval = max(1, n_epochs // 50)
    episodes = []
    avg_rewards = []
    baseline_means = []
    grad_norm_means = []
    entropy_means = []
    lr_means = []
    p_check_means = []
    p_bet_means = []
    p_call_means = []
    p_fold_means = []
    call_rates = []
    bluff_rates = []
    value_bet_rates = []

    for start in range(0, len(df), interval):
        chunk = df.iloc[start:start + interval]
        if chunk.empty:
            continue
        episodes.append(int(chunk["episode"].iloc[-1]))
        avg_rewards.append(chunk["reward"].mean())
        baseline_means.append(chunk["baseline"].mean())
        grad_norm_means.append(chunk["grad_norm"].mean())
        entropy_means.append(chunk["entropy"].mean())
        lr_means.append(chunk["learning_rate"].mean())
        p_check_means.append(chunk["p_check"].mean())
        p_bet_means.append(chunk["p_bet"].mean())
        p_call_means.append(chunk["p_call"].mean())
        p_fold_means.append(chunk["p_fold"].mean())
        call_rates.append(chunk["call_rate"].mean())
        bluff_rates.append(chunk["bluff_rate"].mean())
        value_bet_rates.append(chunk["value_bet_rate"].mean())

    # Summary for last 10% of episodes
    recent_count = max(1, int(len(df) * 0.1))
    recent = df.tail(recent_count)
    avg_reward_last = float(recent["reward"].mean())
    win_rate_last = float((recent["reward"] > 0).mean())
    entropy_last = float(recent["entropy"].mean())

    summary = {
        "average_reward": round(avg_reward_last, 4),
        "win_rate_est": round(win_rate_last, 4),
        "entropy": round(entropy_last, 4),
        "action_rates": {
            "call_rate": round(float(recent["call_rate"].mean()), 4),
            "bluff_rate": round(float(recent["bluff_rate"].mean()), 4),
            "value_bet_rate": round(float(recent["value_bet_rate"].mean()), 4),
        },
        "first_move_probabilities": {
            "check": round(float(recent["p_check"].mean()), 4),
            "bet": round(float(recent["p_bet"].mean()), 4),
            "call": round(float(recent["p_call"].mean()), 4),
            "fold": round(float(recent["p_fold"].mean()), 4),
        },
    }

    with open(os.path.join(run_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    # 1. Average reward over time
    plt.figure()
    plt.plot(episodes, avg_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Average Reward Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(training_dir, "avg_reward.pdf"))
    plt.close()

    # 2. Baseline vs Average Reward
    plt.figure()
    plt.plot(episodes, baseline_means, label="Baseline")
    plt.plot(episodes, avg_rewards, label="Average Reward")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title("Baseline vs Average Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(training_dir, "baseline_vs_reward.pdf"))
    plt.close()

    # 3. Gradient norm
    plt.figure()
    plt.plot(episodes, grad_norm_means)
    plt.xlabel("Episode")
    plt.ylabel("Gradient Norm")
    plt.title("Average Gradient Norm")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(training_dir, "grad_norm.pdf"))
    plt.close()

    # 4. Entropy
    plt.figure()
    plt.plot(episodes, entropy_means)
    plt.xlabel("Episode")
    plt.ylabel("Entropy")
    plt.title("Entropy Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(training_dir, "entropy.pdf"))
    plt.close()

    # 5. Learning rate
    plt.figure()
    plt.plot(episodes, lr_means)
    plt.xlabel("Episode")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(training_dir, "learning_rate.pdf"))
    plt.close()

    # 6. First-move action probabilities
    plt.figure()
    plt.plot(episodes, p_check_means, label="p_check")
    plt.plot(episodes, p_bet_means, label="p_bet")
    plt.plot(episodes, p_call_means, label="p_call")
    plt.plot(episodes, p_fold_means, label="p_fold")
    plt.xlabel("Episode")
    plt.ylabel("Probability")
    plt.title("First-Move Action Probabilities")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(training_dir, "first_move_probs.pdf"))
    plt.close()

    # 7. Strategic action rates
    plt.figure()
    plt.plot(episodes, bluff_rates, label="Bluff rate")
    plt.plot(episodes, value_bet_rates, label="Value bet rate")
    plt.plot(episodes, call_rates, label="Call rate")
    plt.xlabel("Episode")
    plt.ylabel("Rate")
    plt.title("Strategic Action Rates")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "strategic_rates.pdf"))
    plt.close()


def analyze_run(run_dir: str, df: pd.DataFrame | None = None) -> None:
    """Load training data for *run_dir* and generate plots/summary outputs."""
    normalized_df = load_training_dataframe(run_dir, df)
    analyze(normalized_df, run_dir)


def main() -> None:
    """Entry point when running as a script."""
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    analyze_run(run_dir)


if __name__ == "__main__":
    main()
