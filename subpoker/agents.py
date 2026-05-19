import random
from typing import Optional

try:
    from subpoker.engine import KuhnPokerEnv
except ImportError:
    from engine import KuhnPokerEnv


class Agent():
    def act(self, state: dict, legal_actions: list) -> int:
        """Base class for all agents."""
        raise NotImplementedError
    

class NashAgent(Agent):
    """Kuhn poker Nash equilibrium strategy with a parameter ``alpha``.

    The parameter ``alpha`` must be in the interval [0, 1/3]. Actions are
    chosen according to the following pattern:

    Player 1 (first to act)
        Jack  -> bet with probability ``alpha``
        Queen -> always check. If player 2 subsequently bets, call with
                probability ``1/3 + alpha``
        King  -> bet with probability ``3 * alpha``

    Player 2 (facing a bet)
        Jack  -> fold (or check if player 1 checked)
        Queen -> call with probability ``1/3``
        King  -> always call

    When player 1 checks and player 2 is to act, the agent follows the common
    equilibrium continuation where player 2 bets with a King, bets with a Queen
    with probability ``1 - 3 * alpha`` and checks with a Jack.
    """

    def __init__(self, alpha: float = 1/3, random_seed: int | None = None):        
        if not 0 <= alpha <= 1/3:
            raise ValueError("alpha must be between 0 and 1/3")
        self.alpha = alpha
        self._rng = random.Random(random_seed)
        self.name = f"NashAgent(alpha={alpha:.3f})"

    def act(self, state: dict, legal_actions: list) -> int:
        hand = state["hand"]
        history : list[int] = state["history"]

        a = self.alpha

        check = KuhnPokerEnv.CHECK
        bet = KuhnPokerEnv.BET
        call = KuhnPokerEnv.CALL
        fold = KuhnPokerEnv.FOLD


        # Player 1 actions (opening move)
        if history == []:
            if hand == 1:  # Jack
                return bet if self._rng.random() < a else check
            if hand == 2:  # Queen
                return check
            if hand == 3:  # King
                return bet if self._rng.random() < 3 * a else check
        
        # Player 2 after a check from player 1
        if history == [check]:
            if hand == 1:
                return bet if self._rng.random() < a else check
            if hand == 2:
                return bet if self._rng.random() < max(0, 1 - 3 * a) else check
            if hand == 3:
                return bet
    
        # Player 2 facing a bet from player 1
        if history == [bet]:
            if hand == 1:
                return fold
            if hand == 2:
                return call if self._rng.random() < 1/3 else fold
            if hand == 3:
                return call

        # Player 1 responding to a bet after checking
        if history == [check, bet]:
            if hand == 1:
                return fold
            if hand == 2:
                return call if self._rng.random() < (1/3 + a) else fold
            if hand == 3:
                return call

        # Default fall back if no specific rule applies
        if check in legal_actions:
            return check

        return random.choice(legal_actions)
    