"""
Evaluate a saved NumPy policy network from a timestamped run directory.

The script expects one argument: the run directory containing ``config.json``
and ``model.npz``. Testing outputs are written under ``<run_dir>/testing``.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

FILE_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.dirname(FILE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from subpoker.agents import NashAgent
from subpoker.engine import KuhnPokerEnv as Env
from subpoker.numpy_nn import NumNet


def fix_seed(seed: int) -> None:
    """Set the random seed for reproducibility."""
    np.random.seed(seed % (2**32))


SEEDS = [1337, 271828, 314159, 8675309, 29461070, 52960675432, 381420072889]
FIRST_PLAYERS = [0, 1]
TOTAL_TEST_EPISODES = 168000
EPISODES_CYCLES = TOTAL_TEST_EPISODES // (len(SEEDS) * len(FIRST_PLAYERS))

CHECK = Env.CHECK
BET = Env.BET
CALL = Env.CALL
FOLD = Env.FOLD
ACTION_TO_INDEX = {
    CHECK: 0,
    CALL: 1,
    BET: 2,
    FOLD: 3,
}
ACTION_LABELS = {
    CHECK: "check",
    BET: "bet",
    CALL: "call",
    FOLD: "fold",
}
ACTION_VECTOR = np.array([CHECK, CALL, BET, FOLD], dtype=int)
LEGAL_ACTION_MASKS = {
    frozenset((CHECK, BET)): np.array([1.0, 0.0, 1.0, 0.0]),
    frozenset((CALL, FOLD)): np.array([0.0, 1.0, 0.0, 1.0]),
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


def load_model(run_dir: str, config: dict) -> NumNet:
    """Load the saved NumPy model from the run directory."""
    return NumNet.load(os.path.join(run_dir, "model.npz"), config)


def encode_state(state: dict) -> np.ndarray:
    """Encode the game state into the same 12D vector used during training."""
    hand = state["hand"]
    hand_vec = [1.0 if (i + 1) == hand else 0.0 for i in range(3)]
    history = tuple(state["history"])
    history_index = VALID_HISTORIES[history]
    history_vec = [1.0 if i == history_index else 0.0 for i in range(9)]
    return np.array(hand_vec + history_vec, dtype=float)


def sample_action(probs: np.ndarray, legal_actions: list[int]) -> int:
    """Sample an action from *probs* restricted to *legal_actions*."""
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
    """Convert an action history to the CSV string representation."""
    return "-".join(ACTION_LABELS[action] for action in history)


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise SystemExit(f"Usage: {os.path.basename(__file__)} <run_dir>")

    run_dir = args[0]
    testing_dir = os.path.join(run_dir, "testing")
    os.makedirs(testing_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)

    rewards_list: list[int] = []
    hands_log: list[int] = []
    opp_hands_log: list[int] = []
    first_player_log: list[int] = []
    histories_log: list[list[int]] = []

    opening_counts = {
        1: {BET: 0, CHECK: 0},
        2: {BET: 0, CHECK: 0},
        3: {BET: 0, CHECK: 0},
    }
    after_check_counts = {
        1: {BET: 0, CHECK: 0},
        2: {BET: 0, CHECK: 0},
        3: {BET: 0, CHECK: 0},
    }
    facing_open_bet_counts = {
        1: {CALL: 0, FOLD: 0},
        2: {CALL: 0, FOLD: 0},
        3: {CALL: 0, FOLD: 0},
    }
    facing_check_bet_counts = {
        1: {CALL: 0, FOLD: 0},
        2: {CALL: 0, FOLD: 0},
        3: {CALL: 0, FOLD: 0},
    }

    with tqdm(total=EPISODES_CYCLES, desc="Testing") as pbar:
        for seed in SEEDS:
            fix_seed(seed)
            env = Env(seed)
            nn = load_model(run_dir, config)
            agent = NashAgent(alpha=0.333, random_seed=seed)

            for _ in range(EPISODES_CYCLES):
                for fp in FIRST_PLAYERS:
                    env.reset()
                    env.first_player = fp
                    env.current_player = fp
                    done = False

                    while not done:
                        state = env.get_state()
                        legal = env.legal_actions()

                        if state["player"] == 0:
                            x = encode_state(state)
                            probs = nn.forward(x)
                            action = sample_action(probs, legal)
                        else:
                            action = agent.act(state, legal)

                        _, rewards, done, _ = env.step(action)

                    reward = rewards[0]  # type: ignore[index]
                    rewards_list.append(reward)
                    history = env.history.copy()
                    hand = env.hands[0]
                    opp_hand = env.hands[1]

                    hands_log.append(hand)
                    opp_hands_log.append(opp_hand)
                    first_player_log.append(fp)
                    histories_log.append(history)

                    if fp == 0 and history:
                        opening_action = history[0]
                        if opening_action in (BET, CHECK):
                            opening_counts[hand][opening_action] += 1

                        if len(history) >= 3:
                            response_action = history[2]
                            if response_action in (CALL, FOLD):
                                facing_check_bet_counts[hand][response_action] += 1

                    if fp == 1:
                        if history and history[0] == CHECK and len(history) >= 2:
                            after_check_action = history[1]
                            if after_check_action in (BET, CHECK):
                                after_check_counts[hand][after_check_action] += 1

                        if history and history[0] == BET and len(history) >= 2:
                            response_action = history[1]
                            if response_action in (CALL, FOLD):
                                facing_open_bet_counts[hand][response_action] += 1

                    pbar.update(1)

    def pct(a: int, b: int) -> float:
        total = a + b
        return round(100 * a / total, 3) if total else 0.0

    hand_names = {1: "Jack", 2: "Queen", 3: "King"}
    opening_table = {}
    after_check_table = {}
    facing_open_bet_table = {}
    facing_check_bet_table = {}

    for hand, name in hand_names.items():
        opening = opening_counts[hand]
        after_check = after_check_counts[hand]
        facing_open_bet = facing_open_bet_counts[hand]
        facing_check_bet = facing_check_bet_counts[hand]

        opening_table[name] = {
            "bet": pct(opening[BET], opening[CHECK]),
            "check": pct(opening[CHECK], opening[BET]),
        }
        after_check_table[name] = {
            "bet": pct(after_check[BET], after_check[CHECK]),
            "check": pct(after_check[CHECK], after_check[BET]),
        }
        facing_open_bet_table[name] = {
            "call": pct(facing_open_bet[CALL], facing_open_bet[FOLD]),
            "fold": pct(facing_open_bet[FOLD], facing_open_bet[CALL]),
        }
        facing_check_bet_table[name] = {
            "call": pct(facing_check_bet[CALL], facing_check_bet[FOLD]),
            "fold": pct(facing_check_bet[FOLD], facing_check_bet[CALL]),
        }

    rewards_mean = sum(rewards_list) / len(rewards_list) if rewards_list else 0.0
    summary = {
        "average_reward": round(rewards_mean, 4),
        "action_table": {
            "opening_check_bet": opening_table,
            "after_opponent_check_check_bet": after_check_table,
            "facing_open_bet_call_fold": facing_open_bet_table,
            "facing_check_bet_call_fold": facing_check_bet_table,
        },
    }

    with open(os.path.join(testing_dir, "testing_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    avg_series = pd.Series(rewards_list).expanding(200).mean()
    plt.figure()
    plt.plot(avg_series)
    plt.xlabel("Evaluation Episode")
    plt.ylabel("Cumulative Mean Reward")
    plt.title("Cumulative Mean Reward Over Evaluation Order")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(testing_dir, "avg_reward.pdf"))
    plt.close()

    pd.DataFrame({
        "hand": hands_log,
        "opp_hand": opp_hands_log,
        "first_to_act": first_player_log,
        "history": [history_to_text(history) for history in histories_log],
        "reward": rewards_list,
    }).to_csv(os.path.join(testing_dir, "full_testing_data.csv"), index=False)


if __name__ == "__main__":
    main()
