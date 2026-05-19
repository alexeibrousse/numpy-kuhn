"""
Analyzes logged data from a training run of the NumNet agent and generates graphs.
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt

FILE_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.dirname(FILE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from subpoker.utils import parse_episode

CARD_NAMES = {
    "jack": "Jack",
    "queen": "Queen",
    "king": "King",
}

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
    "call_rate": 0.0,
    "bluff_rate": 0.0,
    "value_bet_rate": 0.0,
    "acted_first": 0.0,
    "ended_check_check": 0.0,
    "ended_bet_call": 0.0,
    "ended_bet_fold": 0.0,
    "ended_check_bet_call": 0.0,
    "ended_check_bet_fold": 0.0,
    "has_jack": 0.0,
    "has_queen": 0.0,
    "has_king": 0.0,
}

BOOL_COLUMNS = [
    "acted_first",
    "ended_check_check",
    "ended_bet_call",
    "ended_bet_fold",
    "ended_check_bet_call",
    "ended_check_bet_fold",
    "has_jack",
    "has_queen",
    "has_king",
]


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

    if "call_rate" not in df:
        if "is_call" in df:
            df["call_rate"] = df["is_call"].astype(float)
        elif "history" in df:
            parsed = df.apply(parse_episode, axis=1, result_type="expand")
            parsed.columns = [
                "is_bluff",
                "is_value_bet",
                "is_call",
                "is_fold",
                "has_jack",
                "has_queen",
                "has_king",
            ]
            for col in parsed.columns:
                if col not in df:
                    df[col] = parsed[col]
            df["call_rate"] = df["is_call"].astype(float)

    if "bluff_rate" not in df and "is_bluff" in df:
        df["bluff_rate"] = df["is_bluff"].astype(float)

    if "value_bet_rate" not in df and "is_value_bet" in df:
        df["value_bet_rate"] = df["is_value_bet"].astype(float)

    for col, default in REQUIRED_COLUMNS.items():
        if col not in df:
            df[col] = default

    numeric_cols = [c for c in df.columns if (c in REQUIRED_COLUMNS and c not in BOOL_COLUMNS) or c == "episode"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    for col in BOOL_COLUMNS:
        normalized = df[col].astype(str).str.strip().str.lower()
        df[col] = normalized.isin({"true", "1", "1.0"})

    return df


def _augment_action_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct the NN's realized decisions from the episode outcome so they can
    be aggregated by card and decision context.
    """
    df = df.copy()

    acted_first = df["acted_first"].astype(bool)
    ended_check_check = df["ended_check_check"].astype(bool)
    ended_bet_call = df["ended_bet_call"].astype(bool)
    ended_bet_fold = df["ended_bet_fold"].astype(bool)
    ended_check_bet_call = df["ended_check_bet_call"].astype(bool)
    ended_check_bet_fold = df["ended_check_bet_fold"].astype(bool)

    df["check_bet_available"] = acted_first | (
        ~acted_first & (ended_check_check | ended_check_bet_call | ended_check_bet_fold)
    )
    df["call_fold_available"] = (
        (acted_first & (ended_check_bet_call | ended_check_bet_fold))
        | (~acted_first & (ended_bet_call | ended_bet_fold))
    )

    df["took_check"] = (
        (acted_first & (ended_check_check | ended_check_bet_call | ended_check_bet_fold))
        | (~acted_first & ended_check_check)
    )
    df["took_bet"] = (
        (acted_first & (ended_bet_call | ended_bet_fold))
        | (~acted_first & (ended_check_bet_call | ended_check_bet_fold))
    )
    df["took_call"] = (acted_first & ended_check_bet_call) | (~acted_first & ended_bet_call)
    df["took_fold"] = (acted_first & ended_check_bet_fold) | (~acted_first & ended_bet_fold)

    for key, column in (("jack", "has_jack"), ("queen", "has_queen"), ("king", "has_king")):
        df[f"card_{key}"] = df[column].astype(bool)

    return df


def _safe_rate(df: pd.DataFrame, numerator_col: str, denominator_col: str) -> float:
    subset = df[df[denominator_col]]
    if len(subset) == 0:
        return 0.0
    return float(subset[numerator_col].mean())


def _card_summary(df: pd.DataFrame) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for key in CARD_NAMES:
        card_df = df[df[f"card_{key}"]]
        summary.append({
            "card": key,
            "check_vs_bet": {
                "check": round(_safe_rate(card_df, "took_check", "check_bet_available"), 4),
                "bet": round(_safe_rate(card_df, "took_bet", "check_bet_available"), 4),
            },
            "call_vs_fold": {
                "call": round(_safe_rate(card_df, "took_call", "call_fold_available"), 4),
                "fold": round(_safe_rate(card_df, "took_fold", "call_fold_available"), 4),
            },
        })
    return summary


def _plot_card_action_rates(df: pd.DataFrame, episodes: list[int], interval: int, run_dir: str) -> None:
    for key, label in CARD_NAMES.items():
        check_rates = []
        bet_rates = []
        call_rates = []
        fold_rates = []

        for start in range(0, len(df), interval):
            episode_chunk = df.iloc[start:start + interval]
            if len(episode_chunk) == 0:
                continue
            episode_min = int(episode_chunk["episode"].iloc[0])
            episode_max = int(episode_chunk["episode"].iloc[-1])
            chunk = df[
                df[f"card_{key}"]
                & (df["episode"] >= episode_min)
                & (df["episode"] <= episode_max)
            ]
            check_rates.append(_safe_rate(chunk, "took_check", "check_bet_available"))
            bet_rates.append(_safe_rate(chunk, "took_bet", "check_bet_available"))
            call_rates.append(_safe_rate(chunk, "took_call", "call_fold_available"))
            fold_rates.append(_safe_rate(chunk, "took_fold", "call_fold_available"))

        plt.figure()
        plt.plot(episodes, check_rates, label="Check")
        plt.plot(episodes, bet_rates, label="Bet")
        plt.plot(episodes, call_rates, label="Call")
        plt.plot(episodes, fold_rates, label="Fold")
        plt.xlabel("Episode")
        plt.ylabel("Rate")
        plt.title(f"{label} Action Rates")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"{key}_action_rates.pdf"))
        plt.close()
        


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

    if len(df) == 0:
        raise ValueError("Training log is empty; no data to analyze.")

    df = _augment_action_columns(df)

    n_epochs = len(df)
    interval = max(1, n_epochs // 50)
    episodes = []
    avg_rewards = []
    grad_norm_means = []
    entropy_means = []
    call_rates = []
    bluff_rates = []
    value_bet_rates = []

    for start in range(0, len(df), interval):
        chunk = df.iloc[start:start + interval]
        if len(chunk) == 0:
            continue
        episodes.append(int(chunk["episode"].iloc[-1]))
        avg_rewards.append(chunk["reward"].mean())
        grad_norm_means.append(chunk["grad_norm"].mean())
        entropy_means.append(chunk["entropy"].mean())
        call_rates.append(chunk["call_rate"].mean())
        bluff_rates.append(chunk["bluff_rate"].mean())
        value_bet_rates.append(chunk["value_bet_rate"].mean())

    # Summary for last 10% of episodes
    recent_count = max(1, int(len(df) * 0.1))
    recent = df.tail(recent_count)
    avg_reward_last = float(recent["reward"].mean())
    win_rate_last = float((recent["reward"] > 0).mean())

    summary = {
        "average_reward": round(avg_reward_last, 4),
        "win_rate_est": round(win_rate_last, 4),
        "card_action_probabilities": _card_summary(recent),
        "action_rates": {
            "call_rate": round(float(recent["call_rate"].mean()), 4),
            "bluff_rate": round(float(recent["bluff_rate"].mean()), 4),
            "value_bet_rate": round(float(recent["value_bet_rate"].mean()), 4),
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
    plt.savefig(os.path.join(run_dir, "avg_reward.pdf"))
    plt.close()

    # 2. Gradient norm
    plt.figure()
    plt.plot(episodes, grad_norm_means)
    plt.xlabel("Episode")
    plt.ylabel("Gradient Norm")
    plt.title("Average Gradient Norm")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(training_dir, "grad_norm.pdf"))
    plt.close()

    # 3. Entropy
    plt.figure()
    plt.plot(episodes, entropy_means)
    plt.xlabel("Episode")
    plt.ylabel("Entropy")
    plt.title("Entropy Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(training_dir, "entropy.pdf"))
    plt.close()

    # 4. Card-specific action rates
    _plot_card_action_rates(df, episodes, interval, run_dir)

    # 5. Strategic action rates
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
