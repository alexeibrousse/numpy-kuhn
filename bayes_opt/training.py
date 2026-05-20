"""Reusable NumNet training loop for Optuna trials and final runs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import trange

from subpoker.agents import NashAgent
from bayes_opt.config import (
    TrainingConfig,
    metadata_from_training_config,
)
from subpoker.engine import KuhnPokerEnv as Env
from subpoker.numpy_nn import NumNet
from subpoker.train_analysis import analyze_run
from subpoker.utils import grad_norm, save_metadata


ACTION_TO_INDEX = {
    Env.CHECK: 0,
    Env.CALL: 1,
    Env.BET: 2,
    Env.FOLD: 3,
}

VALID_HISTORIES = {
    (): 0,
    (Env.CHECK,): 1,
    (Env.BET,): 2,
    (Env.CHECK, Env.CHECK): 3,
    (Env.CHECK, Env.BET): 4,
    (Env.BET, Env.CALL): 5,
    (Env.BET, Env.FOLD): 6,
    (Env.CHECK, Env.BET, Env.CALL): 7,
    (Env.CHECK, Env.BET, Env.FOLD): 8,
}

TERMINAL_HISTORY_FLAGS = {
    (Env.CHECK, Env.CHECK): "ended_check_check",
    (Env.BET, Env.CALL): "ended_bet_call",
    (Env.BET, Env.FOLD): "ended_bet_fold",
    (Env.CHECK, Env.BET, Env.CALL): "ended_check_bet_call",
    (Env.CHECK, Env.BET, Env.FOLD): "ended_check_bet_fold",
}


@dataclass
class TrainingResult:
    run_dir: Path
    model_path: Path
    summary: dict
    config: TrainingConfig


def encode_state(state: dict) -> np.ndarray:
    """
    Encodes the game state into a 12-dimensional vector.
    The first three dimensions represent the player's card as one-hot:
        [0, 0, 1] is King
        [0, 1, 0] is Queen
        [1, 0, 0] is Jack
    The remaining nine dimensions encode the current history as a one-hot
    over the 9 valid history patterns defined in `VALID_HISTORIES`.
    """
    hand = state["hand"]
    history = state["history"]

    card_vec = [0, 0, 0]
    card_vec[hand - 1] = 1

    history = tuple(history)
    action_index = VALID_HISTORIES[history]
    history_vec = [1.0 if i == action_index else 0.0 for i in range(9)]

    return np.concatenate([card_vec, history_vec])


def entropy_loss(probs: np.ndarray) -> float:
    return float(-np.sum(probs * np.log(probs + 1e-10)))


def learning_rate_for_episode(config: TrainingConfig, episode: int) -> float:
    if not config.use_learning_rate_decay:
        return config.learning_rate
    return config.learning_rate * (1 - episode / config.n_epochs)


def entropy_coeff_for_episode(config: TrainingConfig, episode: int) -> float:
    if not config.use_entropy_bonus:
        return 0.0
    if not config.use_entropy_decay:
        return config.initial_entropy_coeff
    return config.initial_entropy_coeff * (1 - episode / config.n_epochs)


def update_baseline(config: TrainingConfig, baseline: float, reward: float) -> float:
    if not config.use_baseline:
        return 0.0
    if not 0 < config.baseline_momentum < 1:
        raise ValueError("baseline_momentum must be in ]0, 1[.")

    baseline = baseline * config.baseline_momentum + reward * (1 - config.baseline_momentum)
    if not config.use_baseline_bound:
        return baseline
    if config.baseline_bound <= 0:
        raise ValueError("baseline_bound must be positive.")
    return max(-config.baseline_bound, min(config.baseline_bound, baseline))


def action_probs(env: Env, nn: NumNet, state: dict) -> tuple[int, np.ndarray, np.ndarray, int]:
    x = encode_state(state)
    probs = nn.forward(x)

    legal_actions = env.legal_actions()
    legal_indices = [ACTION_TO_INDEX[action] for action in legal_actions]
    filtered_probs = probs[legal_indices]
    filtered_sum = np.sum(filtered_probs)

    if filtered_sum > 0:
        filtered_probs = filtered_probs / filtered_sum
    else:
        filtered_probs = np.ones_like(filtered_probs) / len(filtered_probs)

    executed_probs = np.zeros_like(probs)
    executed_probs[legal_indices] = filtered_probs
    action = int(np.random.choice(legal_actions, p=filtered_probs))
    action_index = ACTION_TO_INDEX[action]
    return action, x, executed_probs, action_index


def play_episode(
    env: Env,
    nn: NumNet,
    agent: NashAgent,
    config: TrainingConfig,
) -> tuple[int, list[tuple[np.ndarray, int, np.ndarray]], dict]:
    state = env.reset()
    done = False
    reward = 0
    trajectory: list[tuple[np.ndarray, int, np.ndarray]] = []
    first_probs: list[float] | None = None
    entropy_total = 0.0
    entropy_count = 0

    while not done:
        if state["player"] != config.player_number:
            action = agent.act(state, env.legal_actions())
        else:
            action, x, probs, action_index = action_probs(env, nn, state)
            trajectory.append((x, action_index, probs))
            if first_probs is None and not state["history"]:
                first_probs = np.round(probs, 3).tolist()
            entropy_total += entropy_loss(probs)
            entropy_count += 1

        state, step_rewards, done, _ = env.step(action)
        reward = int(step_rewards[config.player_number])

    prob_summary = {
        "first_probs": first_probs or [0.0, 0.0, 0.0, 0.0],
        "entropy": entropy_total / entropy_count if entropy_count else 0.0,
    }
    return reward, trajectory, prob_summary


def clip_gradients(
    config: TrainingConfig,
    dW1: np.ndarray,
    db1: np.ndarray,
    dW2: np.ndarray,
    db2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total_norm = grad_norm([dW1, db1, dW2, db2])
    if total_norm > config.gradient_clip:
        scale = config.gradient_clip / (total_norm + 1e-6)
        dW1 *= scale
        db1 *= scale
        dW2 *= scale
        db2 *= scale
    return dW1, db1, dW2, db2


def update_nn(
    nn: NumNet,
    config: TrainingConfig,
    trajectory: list[tuple[np.ndarray, int, np.ndarray]],
    advantage: float,
    entropy_coeff: float,
) -> float:
    if not trajectory:
        return 0.0

    dW1 = np.zeros_like(nn.W1)
    db1 = np.zeros_like(nn.b1)
    dW2 = np.zeros_like(nn.W2)
    db2 = np.zeros_like(nn.b2)

    for x, action_index, probs in trajectory:
        step_advantage = advantage
        if config.use_entropy_bonus:
            step_advantage += entropy_coeff * entropy_loss(probs)

        nn.forward(x)
        gW1, gb1, gW2, gb2 = nn.backward(action_index, step_advantage, probs)
        dW1 += gW1
        db1 += gb1
        dW2 += gW2
        db2 += gb2

    dW1 /= len(trajectory)
    db1 /= len(trajectory)
    dW2 /= len(trajectory)
    db2 /= len(trajectory)

    if config.use_gradient_clipping:
        dW1, db1, dW2, db2 = clip_gradients(config, dW1, db1, dW2, db2)

    gradient_norm = float(np.sqrt(np.sum(dW1**2) + np.sum(db1**2) + np.sum(dW2**2) + np.sum(db2**2)))
    nn.update(dW1, db1, dW2, db2)
    return gradient_norm


def episode_flags(
    history: list[int],
    hand: int,
    first_player: int,
    player_number: int,
) -> dict[str, bool]:
    flags = {
        "is_bluff": False,
        "is_value_bet": False,
        "is_call": False,
        "is_fold": False,
        "has_jack": hand == 1,
        "has_queen": hand == 2,
        "has_king": hand == 3,
        "acted_first": first_player == player_number,
        "ended_check_check": False,
        "ended_bet_call": False,
        "ended_bet_fold": False,
        "ended_check_bet_call": False,
        "ended_check_bet_fold": False,
    }

    terminal_flag = TERMINAL_HISTORY_FLAGS.get(tuple(history))
    if terminal_flag is not None:
        flags[terminal_flag] = True

    if history:
        last_idx = len(history) - 1
        actor_last = (first_player + last_idx) % 2
        last_action = history[-1]
        if actor_last == player_number and last_action in (Env.CALL, Env.FOLD):
            flags["is_call"] = last_action == Env.CALL
            flags["is_fold"] = last_action == Env.FOLD

        if len(history) >= 2:
            bet_idx = len(history) - 2
            actor_bet = (first_player + bet_idx) % 2
            if actor_bet == player_number and history[bet_idx] == Env.BET:
                flags["is_bluff"] = hand in (1, 2)
                flags["is_value_bet"] = hand == 3
    return flags


def episode_log_row(
    env: Env,
    nn: NumNet,
    config: TrainingConfig,
    episode: int,
    reward: int,
    baseline: float,
    gradient_norm: float,
    prob_summary: dict,
) -> dict:
    first_probs = prob_summary["first_probs"]
    hand = env.hands[config.player_number]
    first_player = env.first_to_start()
    return {
        "episode": episode,
        "reward": reward,
        "baseline": baseline,
        "learning_rate": nn.lr,
        "grad_norm": gradient_norm,
        "entropy": float(prob_summary["entropy"]),
        "p_check": first_probs[0],
        "p_call": first_probs[1],
        "p_bet": first_probs[2],
        "p_fold": first_probs[3],
        **episode_flags(env.history, hand, first_player, config.player_number),
    }


def summarize_training(rows: list[dict]) -> dict:
    if not rows:
        return {
            "average_reward": 0.0,
            "win_rate_est": 0.0,
            "action_rates": {
                "call_rate": 0.0,
                "bluff_rate": 0.0,
                "value_bet_rate": 0.0,
            },
        }

    rewards = np.array([row["reward"] for row in rows], dtype=float)
    return {
        "average_reward": round(float(rewards.mean()), 4),
        "win_rate_est": round(float((rewards > 0).mean()), 4),
        "action_rates": {
            "call_rate": round(float(np.mean([row["is_call"] for row in rows])), 4),
            "bluff_rate": round(float(np.mean([row["is_bluff"] for row in rows])), 4),
            "value_bet_rate": round(float(np.mean([row["is_value_bet"] for row in rows])), 4),
        },
    }


def train_policy(
    config: TrainingConfig,
    run_dir: Path,
    save_full_training: bool,
    run_analysis: bool,
    show_progress: bool = True,
) -> TrainingResult:
    run_dir.mkdir(parents=True, exist_ok=True)
    training_dir = run_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(config.random_seed % (2**32))
    env = Env(config.random_seed)
    nn = NumNet(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        output_size=config.output_size,
        learning_rate=config.learning_rate,
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        epsilon=config.adam_epsilon,
    )
    agent = NashAgent(alpha=config.nash_alpha, random_seed=config.random_seed)

    metadata = metadata_from_training_config(config)
    save_metadata(metadata, str(run_dir))

    baseline = 0.0
    full_rows: list[dict] = []
    recent_rows: list[dict] = []
    recent_count = max(1, int(config.n_epochs * 0.1))
    recent_start = config.n_epochs - recent_count + 1

    for episode in trange(1, config.n_epochs + 1, desc=f"Training {run_dir.name}"):
        reward, trajectory, prob_summary = play_episode(env, nn, agent, config)
        advantage = reward - baseline
        fixed_baseline = baseline
        baseline = update_baseline(config, baseline, reward)
        nn.lr = learning_rate_for_episode(config, episode)
        entropy_coeff = entropy_coeff_for_episode(config, episode)
        gradient_norm = update_nn(nn, config, trajectory, advantage, entropy_coeff)

        if save_full_training or episode >= recent_start:
            row = episode_log_row(
                env,
                nn,
                config,
                episode,
                reward,
                fixed_baseline,
                gradient_norm,
                prob_summary,
            )
            if save_full_training:
                full_rows.append(row)
            if episode >= recent_start:
                recent_rows.append(row)

    if save_full_training:
        full_df = pd.DataFrame(full_rows)
        full_df.to_csv(training_dir / "full_training_data.csv", index=False)
        if run_analysis:
            analyze_run(str(run_dir), full_df)

    summary = summarize_training(recent_rows)
    with open(run_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    model_path = run_dir / "model.npz"
    nn.save(str(model_path))
    return TrainingResult(run_dir=run_dir, model_path=model_path, summary=summary, config=config)
