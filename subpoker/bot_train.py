"""Train a simple policy network for Kuhn poker using NumNet."""

import random
import numpy as np
import pandas as pd
import os
import sys
import subprocess
from tqdm import trange

FILE_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.dirname(FILE_DIR)
if FILE_DIR not in sys.path:
    sys.path.append(FILE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)


from subpoker.utils import create_run_dir, save_metadata, grad_norm
from subpoker.engine import KuhnPokerEnv as Env
from subpoker.agents import NashAgent
from subpoker.numpy_nn import NumNet



# ————— Environment and reproducibility ————— #

random_seed = random.randint(0, 2**32 - 1)
# random_seed = 1906220402
np.random.seed(random_seed)
env = Env(random_seed)
player_number = 0

"""
Seeds for reproducibility:
1. 525518843
2. 2342489760
3. 2097210685
4. 1906220402
"""



# ————— Hyperparameters ————— #

n_epochs = 500000
learning_rate = 5e-6
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_epsilon = 1e-8
nn = NumNet(
    input_size=12,
    hidden_size=20,
    output_size=4,
    learning_rate=learning_rate,
    beta1=adam_beta1,
    beta2=adam_beta2,
    epsilon=adam_epsilon,
)

agent = NashAgent()
initial_lr = nn.lr
use_learning_rate_decay = True
use_baseline = True
baseline_momentum = 0.10
use_baseline_bound = True
baseline_bound = 15
use_entropy_bonus = False
initial_entropy_coeff = 0.01
use_entropy_decay = True
entropy_coeff = initial_entropy_coeff if use_entropy_bonus else 0.0
use_gradient_clipping = True
gradient_clip = 7.0



# ————— Metadata ————— #

metadata = {
    "implementation": "numpy",
    "agent": agent.name,
    "optimizer": "adam",
    "input_size": nn.input_size,
    "hidden_size": nn.hidden_size,
    "output_size": nn.output_size,
    "initial_learning_rate": initial_lr,
    "adam_beta1": nn.beta1,
    "adam_beta2": nn.beta2,
    "adam_epsilon": nn.epsilon,
    "use_learning_rate_decay": use_learning_rate_decay,
    "activation": "ReLU",
    "number_epochs": n_epochs,
    "use_baseline": use_baseline,
    "baseline_momentum": baseline_momentum,
    "use_baseline_bound": use_baseline_bound,
    "baseline_bound": baseline_bound,
    "use_entropy_bonus": use_entropy_bonus,
    "initial_entropy_coeff": initial_entropy_coeff,
    "use_entropy_decay": use_entropy_decay,
    "random_seed": random_seed,
    "use_gradient_clipping": use_gradient_clipping,
    "gradient_clip": gradient_clip,
}   



# ————— Training helper functions ————— #

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

    # One-hot encoding of the player's hand
    card_vec = [0, 0, 0]
    card_vec[hand - 1] = 1
    
    history = tuple(state["history"])
    action_index = VALID_HISTORIES[history]
    history_vec = [1.0 if i == action_index else 0.0 for i in range(9)]


    return np.concatenate([card_vec, history_vec]) # History of ongoing round



def action_probs(state: dict) -> tuple[int, np.ndarray, np.ndarray, int]:
    """ Calculates the action probabilities for the given state. """
    X = encode_state(state)
    probs = nn.forward(X)
    
    legal_actions = env.legal_actions()
    legal_indices = [ACTION_TO_INDEX[i] for i in legal_actions]

    filtered_probs = probs[legal_indices]
    filtered_sum = np.sum(filtered_probs)

    if filtered_sum > 0:
        filtered_probs /= filtered_sum
    else:
        # Uniform probability over legal actions if all logits are zero
        filtered_probs = np.ones_like(filtered_probs) / len(filtered_probs)
        
    executed_probs = np.zeros_like(probs)
    executed_probs[legal_indices] = filtered_probs
    
    action = np.random.choice(legal_actions, p=filtered_probs)
    action_index = ACTION_TO_INDEX[action]

    return action, X, executed_probs, action_index



def update_baseline(baseline: float, reward: float) -> float:
    """
    Updates the baseline to reduce variance, bounded in [-bound, bound]. 
    """
    if not use_baseline:
        return 0.0

    momentum, bound = baseline_momentum, baseline_bound

    if not (0 < momentum < 1): # If momentum is 0, there is no baseline. If momentum is 1, the baseline is the reward.
        raise ValueError("Momentum must be in ]0, 1[.")

    baseline = baseline * momentum + reward * (1 - momentum)

    if not use_baseline_bound:
        return baseline

    if bound <= 0:
        raise ValueError("Bound must be positive.")

    return max(-bound, min(bound, baseline))  # Ensure baseline is within bounds



def update_advantage(baseline: float, reward: float) -> float:
    """
    Computes the advantage. Can't explain better.
    """
    return reward - baseline



def learning_rate_decay(episode: int) -> float:
    """
    Returns the learning rate for the current episode.
    """  
    if not use_learning_rate_decay:
        return initial_lr
    return initial_lr * (1 - episode / n_epochs)



def entropy_coeff_decay(episode: int) -> float:
    """
    Returns the entropy coefficient for the current episode.
    """
    if not use_entropy_bonus:
        return 0.0
    if not use_entropy_decay:
        return initial_entropy_coeff
    return initial_entropy_coeff * (1 - episode / n_epochs)



def entropy_loss(probs: np.ndarray) -> float:
    """
    Computes the entropy loss for the given probabilities.
    The addition of 1e-10 is to avoid log(0)
    """  
    return -np.sum(probs * np.log(probs + 1e-10))



def clip_gradients(dW1: np.ndarray, db1: np.ndarray, dW2: np.ndarray, db2: np.ndarray, max_norm: float = gradient_clip) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Scales gradients so their global norm does not exceed 'max_norm'.
    """
    total_norm = grad_norm([dW1, db1, dW2, db2])
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        dW1 *= scale
        db1 *= scale
        dW2 *= scale
        db2 *= scale
    return dW1, db1, dW2, db2


def episode_flags(history: list[int], hand: int, first_player: int) -> dict[str, bool]:
    """Compute boolean training metrics directly from the finished episode."""
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
                if hand in (1, 2):
                    flags["is_bluff"] = True
                elif hand == 3:
                    flags["is_value_bet"] = True
    return flags



def step(state: dict, collect_probs: bool = False) -> tuple:
    """
    Plays a round (episode) of the game, alternating between the agent and the neural network.
    """
    done = False
    trajectory: list[tuple[np.ndarray, int, np.ndarray]] = []
    reward: int = 0
    first_probs: list[float] | None = None
    entropy_total = 0.0
    entropy_count = 0

    while not done:
        if state["player"] != player_number: # Agent's turn to play
            legal = env.legal_actions()
            action = agent.act(state, legal)
        
        else: # Neural network's turn to play
            action, X, probs, action_index = action_probs(state)
            trajectory.append((X, action_index, probs))
            if collect_probs:
                rounded_probs = np.round(probs, 3).tolist()
                # Log the bot's opening policy only when it is first to act.
                if first_probs is None and not state["history"]:
                    first_probs = rounded_probs
                entropy_total += float(entropy_loss(np.array(probs)))
                entropy_count += 1
        
        state, step_rewards, done, _ = env.step(action)
        reward = step_rewards[player_number]

    # Round has finished.

    if collect_probs:
        prob_summary = {
            "first_probs": first_probs or [0.0, 0.0, 0.0, 0.0],
            "entropy": entropy_total / entropy_count if entropy_count else 0.0,
        }
        return state, reward, done, trajectory, prob_summary
    else:
        return state, reward, done, trajectory



def update_nn(trajectory: list[tuple[np.ndarray, int, np.ndarray]], advantage: float) -> float:
    """
    This updates the neural network weights and biases based on the trajectory of this episode.
    Returns the norm of the weight gradients.
    """
    dW1 = np.zeros_like(nn.W1)
    db1 = np.zeros_like(nn.b1)
    dW2 = np.zeros_like(nn.W2)
    db2 = np.zeros_like(nn.b2)

    if not trajectory:
        return 0.0

    for X, action_index, probs in trajectory:
        entropy = entropy_loss(probs)
        step_advantage = advantage
        if use_entropy_bonus:
            step_advantage += entropy_coeff * entropy

        nn.forward(X)
        gW1, gb1, gW2, gb2 = nn.backward(action_index, step_advantage, probs)
        dW1 += gW1
        db1 += gb1
        dW2 += gW2
        db2 += gb2
    
    dW1 /= len(trajectory) # Average gradient per episode
    db1 /= len(trajectory)
    dW2 /= len(trajectory)
    db2 /= len(trajectory)

    if use_gradient_clipping:
        dW1, db1, dW2, db2 = clip_gradients(dW1, db1, dW2, db2) 

    grad_norm = np.sqrt(np.sum(dW1**2) + np.sum(db1**2) + np.sum(dW2**2) + np.sum(db2**2))
    nn.update(dW1, db1, dW2, db2)

    return grad_norm





# ————— Utils and data logging ————— #



def data_log(episode_data: list[dict], episode: int, reward: int, baseline: float, grad_norm: float,
 prob_summary: dict) -> None:
    """
    Stores the data of the current episode into a list.
    """
    first_probs = prob_summary["first_probs"]
    episode_entropy = float(prob_summary["entropy"])
    hand = env.hands[player_number]
    first_player = env.first_to_start()
    flags = episode_flags(env.history, hand, first_player)

    episode_data.append({
        "episode": episode,
        "reward": reward,
        "baseline": baseline,
        "learning_rate": nn.lr,
        "grad_norm": grad_norm,
        "entropy": episode_entropy,
        "p_check": first_probs[0],
        "p_call": first_probs[1],
        "p_bet": first_probs[2],
        "p_fold": first_probs[3],
        **flags,
    })





# ————— Main training loop ————— #

def main() -> None:
    """
    Main training loop for the neural network.
    """
    save_metadata(metadata, RUN_DIR)
    baseline = 0.0 # Initial baseline
    global entropy_coeff
    entropy_coeff = initial_entropy_coeff if use_entropy_bonus else 0.0
    state = env.reset() # Initial state of the game
    episode_data: list[dict] = [] # Stores data for each episode, to be analyzed by data_analysis.py
    training_dir = os.path.join(RUN_DIR, "training")
    os.makedirs(training_dir, exist_ok=True)

    for e in trange(1, n_epochs + 1, desc="Training"):
        prob_summary = {}  # Collect only the probability metrics needed for analysis

        state, reward, done, trajectory, prob_summary = step(state, collect_probs=True)
        advantage = update_advantage(baseline, reward)
        fixed_baseline = baseline
        baseline = update_baseline(baseline, reward)
        nn.lr = learning_rate_decay(e)
        entropy_coeff = entropy_coeff_decay(e)
        grad_norm = update_nn(trajectory, advantage)
        if done:
            data_log(episode_data, e, reward, fixed_baseline, grad_norm, prob_summary)
            state = env.reset()
    
    df = pd.DataFrame(episode_data)
    df.to_csv(os.path.join(training_dir, "full_training_data.csv"), index=False)


if __name__ == "__main__":
    RUN_DIR = create_run_dir("bot_train")
    main()
    nn.save(os.path.join(RUN_DIR, "model.npz"))
    print("1/3 - Training completed.")
    analysis_script = os.path.join(FILE_DIR, "train_analysis.py")
    subprocess.run([sys.executable, analysis_script, RUN_DIR], check=True)
    print("2/3 - Analysis completed.")
    testing_script = os.path.join(FILE_DIR, "bot_test.py")
    subprocess.run([sys.executable, testing_script, RUN_DIR], check=True)
    print("3/3 - Testing completed.")
