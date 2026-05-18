import random
from typing import Optional


class KuhnPokerEnv:
    def __init__(self, seed: Optional[int] = None):
        """
        The optional seed parameter is used for reproducibility.
        If not provided, a random seed is used.
        """
        self._rng = random.Random(seed)
        self.reset()
    


    def reset(self):
        """Shuffle the deck, deal cards and return the initial game state."""
        self.deck = [1, 2, 3] # Cards J, Q, K
        self._rng.shuffle(self.deck)    
           
        self.hands = [self.deck[0], self.deck[1]]
        self.pot = 2
        self.bets = [1, 1]
        self.history = []
        self.first_player = self._rng.choice([0,1])
        self.current_player = self.first_player
        self.terminal = False
        self.winner = None

        return self.get_state()
    


    CHECK = 0
    BET = 1
    CALL = 2
    FOLD = 3


    def first_to_start(self) -> int:
        """
        Returns the index of the first player to act.
        """
        return self.first_player



    def legal_actions(self) -> list:
        """ 
        Returns list of legal actions. 
        There are only four cases possible.
        If called at the end or when not supposed to, returns an empty list.
        """

        mapping = {
            tuple(): [self.CHECK, self.BET],
            (self.CHECK,): [self.CHECK, self.BET],
            (self.BET,): [self.CALL, self.FOLD],
            (self.CHECK, self.BET): [self.CALL, self.FOLD],
        }
        return mapping.get(tuple(self.history), [])
    


    def step(self, action: int) -> tuple:
        """
        Applies action to current player.
        Returns (state, reward, done, info)
        """
        if action not in self.legal_actions():
            raise ValueError(f"Invalid action: {action}")
        self.history.append(action)

        # Apply action
        if action == self.CHECK:
            if self.history == [self.CHECK, self.CHECK]:
                self.terminal = True
            else:
                self.current_player = 1 - self.current_player # Switch player
            
        elif action == self.BET:
            self.bets[self.current_player] += 1
            self.pot += 1
            self.current_player = 1 - self.current_player
        
        elif action == self.CALL:
            self.bets[self.current_player] += 1
            self.pot += 1
            self.terminal = True
        
        elif action == self.FOLD:
            self.terminal = True
            self.winner = 1 - self.current_player # Other player wins
        
        # Check if showdown is needed
        if self.terminal and self.winner is None:
            self.winner = 0 if self.hands[0] > self.hands[1] else 1

        done = self.terminal  # True if game is over
        rewards = [0, 0]

        # Assign rewards once game is over
        if done:
            rewards = self.get_reward()      # type: ignore
        
        return self.get_state(), rewards, done, {}
    


    def get_reward(self) -> list:
        """Return net payoff for both players once the game ends."""
        if not self.terminal:
            return [0, 0]

        rewards = [0, 0]
        for i in (0, 1):
            if i == self.winner:
                rewards[i] = self.pot - self.bets[i]
            else:
                rewards[i] = -self.bets[i]
        return rewards    


    
    def is_terminal(self) -> bool:
        """
        Returns True if the game is over.
        """
        return self.terminal
    


    def get_state(self) -> dict:
        """
        Return a dict representing the current observable state for the active player:
          - 'player': current player index (0 or 1)
          - 'hand': that player's private card (1–3)
          - 'history': action sequence so far
          - 'pot': current pot size
          - 'bets': each player's additional bets this round
        """
        return {
            'player': self.current_player,
            'hand': self.hands[self.current_player],
            'history': list(self.history),
            'pot': self.pot,
            'bets': list(self.bets),
            'terminal': self.terminal
        }
    


    def get_state_full(self) -> dict:
        """
        Returns full state including both hands.
        """
        return {
            'hands': list(self.hands),
            'player': self.current_player,
            'history': list(self.history),
            'pot': self.pot,
            'bets': list(self.bets),
            'terminal': self.terminal
        }

