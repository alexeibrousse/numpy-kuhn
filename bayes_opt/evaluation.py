"""Reusable evaluation loop for trained NumNet policies."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import trange

from subpoker.agents import NashAgent
from bayes_opt.config import TestingConfig
from subpoker.engine import KuhnPokerEnv as Env
from subpoker.numpy_nn import NumNet


CHECK = Env.CHECK
BET = Env.BET
CALL = Env.CALL
FOLD = Env.FOLD
ACTION_LABELS = {
    CHECK: "check",
    BET: "bet",
    CALL: "call",
    FOLD: "fold",
}
ACTION_VECTOR = np.array([CHECK, CALL, BET, FOLD], dtype=int)
LEGAL_ACTION_MASKS = {
    frozenset((CHECK, BET)): np.array([1, 0, 1, 0], dtype=int),
    frozenset((CALL, FOLD)): np.array([0, 1, 0, 1], dtype=int),
}

VALID_HISTORIES = {
    (): 0,
    (CHECK,): 1,
    (BET,): 2,
    (CHECK, CHECK): 3,
    (CHECK, BET): 4,
    (BET, CALL): 5,
    (BET, FOLD): 6,
    (CHECK, BET, CALL): 7,
    (CHECK, BET, FOLD): 8,
}


def fix_seed(seed: int) -> None:
    np.random.seed(seed % (2**32))


def load_model(run_dir: Path, config: dict) -> NumNet:
    return NumNet.load(str(run_dir / "model.npz"), config)


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
    history_vec = [1 if i == action_index else 0 for i in range(9)]

    return np.concatenate([card_vec, history_vec])


def sample_action(probs: np.ndarray, legal_actions: list[int]) -> int:
    mask = LEGAL_ACTION_MASKS[frozenset(legal_actions)]
    masked = probs * mask
    total = masked.sum()
    if total == 0:
        masked = mask / mask.sum()
    else:
        masked = masked / total
    index = np.random.choice(len(ACTION_VECTOR), p=masked)
    return int(ACTION_VECTOR[index])


def history_to_text(history: list[int]) -> str:
    return "-".join(ACTION_LABELS[action] for action in history)


def history_to_id(history: list[int]) -> int:
    return VALID_HISTORIES[tuple(history)]


def compact_number(value: float, digits: int = 3) -> int | float:
    rounded = round(value, digits)
    if float(rounded).is_integer():
        return int(rounded)
    return rounded


def _pct(a: int, b: int) -> float:
    total = a + b
    return compact_number(100 * a / total) if total else 0


def evaluate_run(
    run_dir: Path,
    testing_config: TestingConfig,
    *,
    write_full_data: bool,
    write_plot: bool,
    show_progress: bool = True,
) -> dict:
    testing_dir = run_dir / "testing"
    testing_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    rewards_list: list[int] = []
    hands_log: list[int] = []
    opp_hands_log: list[int] = []
    first_player_log: list[int] = []
    histories_log: list[list[int]] = []

    opening_counts = {1: {BET: 0, CHECK: 0}, 2: {BET: 0, CHECK: 0}, 3: {BET: 0, CHECK: 0}}
    after_check_counts = {1: {BET: 0, CHECK: 0}, 2: {BET: 0, CHECK: 0}, 3: {BET: 0, CHECK: 0}}
    facing_open_bet_counts = {1: {CALL: 0, FOLD: 0}, 2: {CALL: 0, FOLD: 0}, 3: {CALL: 0, FOLD: 0}}
    facing_check_bet_counts = {1: {CALL: 0, FOLD: 0}, 2: {CALL: 0, FOLD: 0}, 3: {CALL: 0, FOLD: 0}}

    cycle_size = len(testing_config.seeds) * len(testing_config.first_players)
    cycles = max(1, testing_config.total_episodes // cycle_size)
    actual_episodes = cycles * cycle_size

    per_seed_episodes = cycles * len(testing_config.first_players)
    initial_seed = testing_config.seeds[0]
    fix_seed(initial_seed)
    env = Env(initial_seed)
    nn = load_model(run_dir, config)
    agent = NashAgent(alpha=testing_config.nash_alpha, random_seed=initial_seed)
    for episode_idx in trange(
        actual_episodes,
        desc=f"Testing {run_dir.name}",
        disable=not show_progress,
    ):
        seed_idx, offset = divmod(episode_idx, per_seed_episodes)
        _, fp_idx = divmod(offset, len(testing_config.first_players))
        seed = testing_config.seeds[seed_idx]
        fp = testing_config.first_players[fp_idx]

        if offset == 0:
            fix_seed(seed)
            env = Env(seed)
            nn = load_model(run_dir, config)
            agent = NashAgent(alpha=testing_config.nash_alpha, random_seed=seed)

        env.reset()
        env.first_player = fp
        env.current_player = fp
        done = False
        rewards = [0, 0]

        while not done:
            state = env.get_state()
            legal = env.legal_actions()

            if state["player"] == testing_config.player_number:
                x = encode_state(state)
                probs = nn.forward(x)
                action = sample_action(probs, legal)
            else:
                action = agent.act(state, legal)

            _, rewards, done, _ = env.step(action)

        reward = int(rewards[testing_config.player_number])
        rewards_list.append(reward)
        history = env.history.copy()
        hand = env.hands[testing_config.player_number]
        opp_hand = env.hands[1 - testing_config.player_number]

        if write_full_data:
            hands_log.append(hand)
            opp_hands_log.append(opp_hand)
            first_player_log.append(fp)
            histories_log.append(history)

        if fp == testing_config.player_number and history:
            opening_action = history[0]
            if opening_action in (BET, CHECK):
                opening_counts[hand][opening_action] += 1
            if len(history) >= 3:
                response_action = history[2]
                if response_action in (CALL, FOLD):
                    facing_check_bet_counts[hand][response_action] += 1

        if fp != testing_config.player_number:
            if history and history[0] == CHECK and len(history) >= 2:
                after_check_action = history[1]
                if after_check_action in (BET, CHECK):
                    after_check_counts[hand][after_check_action] += 1
            if history and history[0] == BET and len(history) >= 2:
                response_action = history[1]
                if response_action in (CALL, FOLD):
                    facing_open_bet_counts[hand][response_action] += 1

    opening_table = {}
    after_check_table = {}
    facing_open_bet_table = {}
    facing_check_bet_table = {}

    for hand in (1, 2, 3):
        opening = opening_counts[hand]
        after_check = after_check_counts[hand]
        facing_open_bet = facing_open_bet_counts[hand]
        facing_check_bet = facing_check_bet_counts[hand]

        opening_table[hand] = {
            "bet": _pct(opening[BET], opening[CHECK]),
            "check": _pct(opening[CHECK], opening[BET]),
        }
        after_check_table[hand] = {
            "bet": _pct(after_check[BET], after_check[CHECK]),
            "check": _pct(after_check[CHECK], after_check[BET]),
        }
        facing_open_bet_table[hand] = {
            "call": _pct(facing_open_bet[CALL], facing_open_bet[FOLD]),
            "fold": _pct(facing_open_bet[FOLD], facing_open_bet[CALL]),
        }
        facing_check_bet_table[hand] = {
            "call": _pct(facing_check_bet[CALL], facing_check_bet[FOLD]),
            "fold": _pct(facing_check_bet[FOLD], facing_check_bet[CALL]),
        }

    rewards_mean = sum(rewards_list) / len(rewards_list) if rewards_list else 0.0
    summary = {
        "average_reward": compact_number(rewards_mean, digits=4),
        "actual_episodes": actual_episodes,
        "requested_episodes": testing_config.total_episodes,
        "action_table": {
            "opening_check_bet": opening_table,
            "after_opponent_check_check_bet": after_check_table,
            "facing_open_bet_call_fold": facing_open_bet_table,
            "facing_check_bet_call_fold": facing_check_bet_table,
        },
    }

    with open(testing_dir / "testing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    if write_plot:
        min_periods = min(200, len(rewards_list)) or 1
        avg_series = pd.Series(rewards_list).expanding(min_periods).mean()
        plt.figure()
        plt.plot(avg_series)
        plt.xlabel("Evaluation Episode")
        plt.ylabel("Cumulative Mean Reward")
        plt.title("Cumulative Mean Reward Over Evaluation Order")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(testing_dir / "avg_reward.pdf")
        plt.close()

    if write_full_data:
        pd.DataFrame({
            "hand": hands_log,
            "opp_hand": opp_hands_log,
            "first_to_act": first_player_log,
            "history": [history_to_id(history) for history in histories_log],
            "history_text": [history_to_text(history) for history in histories_log],
            "reward": rewards_list,
        }).to_csv(testing_dir / "full_testing_data.csv", index=False)

    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Bayes-opt NumNet run.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--test-episodes", type=int, default=168_000)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)

    summary = evaluate_run(
        args.run_dir,
        TestingConfig(total_episodes=args.test_episodes),
        write_full_data=True,
        write_plot=True,
        show_progress=not args.no_progress,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
