import os
from datetime import datetime
import json
import math
import pandas as pd

def create_run_dir(subdir: str) -> str:
    project_folder = os.path.dirname(os.path.dirname(__file__))
    base_dir = os.path.join(project_folder, "data", subdir) if subdir else os.path.join(project_folder, "data")
    os.makedirs(base_dir, exist_ok=True)

    run_name = datetime.now().strftime("%d-%m-%y_%H-%M")
    run_dir = os.path.join(base_dir, run_name)

    os.makedirs(run_dir, exist_ok=True)

    return run_dir


def save_metadata(metadata: dict, run_dir: str) -> None:
    """Save run metadata as JSON in *run_dir*."""
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def steps_to_threshold(start_value: float, threshold: float, decay_rate: float) -> int:
    n = math.log(threshold / start_value) / math.log(decay_rate)
    return math.ceil(n)



def parse_episode(row: pd.Series) -> tuple[bool, bool, bool, bool, bool, int, int, int]:
    """Extract common metrics from a logged episode row."""
    history = row.get("history", "")
    actions = history.split("-") if isinstance(history, str) and history else []
    first = int(row.get("first_to_act", 0))
    hand = int(row.get("hand", 0))

    bluff = False
    value_bet = False
    call = False
    fold = False
    responded = False

    if actions:
        last_idx = len(actions) - 1
        actor_last = (first + last_idx) % 2
        last_action = actions[-1]
        if actor_last == 0 and last_action in ("call", "fold"):
            responded = True
            if last_action == "call":
                call = True
            elif last_action == "fold":
                fold = True

        if len(actions) >= 2:
            bet_idx = len(actions) - 2
            actor_bet = (first + bet_idx) % 2
            if actor_bet == 0 and actions[bet_idx] == "bet":
                if hand in (1, 2):
                    bluff = True
                elif hand == 3:
                    value_bet = True

    return bluff, value_bet, call, fold, responded, hand == 1, hand == 2, hand == 3