"""
Individual (Ant) representation for ACO algorithm.
Each individual is a binary vector selecting indicators for entry/exit rules.
"""
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Individual:
    """
    Represents a single ant's solution in the ACO algorithm.

    Binary vector structure:
    - First n_indicators bits: Entry rule indicators
    - Second n_indicators bits: Exit rule indicators

    Example for 83 indicators (166 total bits):
    [entry_0, entry_1, ..., entry_82, exit_0, exit_1, ..., exit_82]
    """
    n_indicators: int = 83
    vector: np.ndarray = field(default=None)
    fitness: Optional[float] = None
    entry_indicators: List[str] = field(default_factory=list)
    exit_indicators: List[str] = field(default_factory=list)
    iteration: int = 0
    index: int = 0
    strategy_name: str = ""
    backtest_result: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.vector is None:
            self.vector = np.zeros(self.n_indicators * 2, dtype=np.int8)
        self.strategy_name = f"ACO_{self.iteration}_{self.index}"

    @property
    def entry_vector(self) -> np.ndarray:
        """Get the entry portion of the binary vector."""
        return self.vector[:self.n_indicators]

    @property
    def exit_vector(self) -> np.ndarray:
        """Get the exit portion of the binary vector."""
        return self.vector[self.n_indicators:]

    @property
    def n_entry_selected(self) -> int:
        """Number of selected entry indicators."""
        return int(np.sum(self.entry_vector))

    @property
    def n_exit_selected(self) -> int:
        """Number of selected exit indicators."""
        return int(np.sum(self.exit_vector))

    def decode(self, indicator_names: List[str]) -> None:
        """
        Decode binary vector to indicator name lists.

        Args:
            indicator_names: List of indicator names in order matching the vector
        """
        self.entry_indicators = [
            name for i, name in enumerate(indicator_names)
            if self.vector[i] == 1
        ]
        self.exit_indicators = [
            name for i, name in enumerate(indicator_names)
            if self.vector[self.n_indicators + i] == 1
        ]

    def encode(self, indicator_names: List[str],
               entry_indicators: List[str],
               exit_indicators: List[str]) -> None:
        """
        Encode indicator lists to binary vector.

        Args:
            indicator_names: Full list of indicator names
            entry_indicators: Selected entry indicator names
            exit_indicators: Selected exit indicator names
        """
        name_to_idx = {name: i for i, name in enumerate(indicator_names)}

        self.vector = np.zeros(self.n_indicators * 2, dtype=np.int8)

        for name in entry_indicators:
            if name in name_to_idx:
                self.vector[name_to_idx[name]] = 1

        for name in exit_indicators:
            if name in name_to_idx:
                self.vector[self.n_indicators + name_to_idx[name]] = 1

        self.entry_indicators = entry_indicators
        self.exit_indicators = exit_indicators

    def is_valid(self, min_entry: int = 2, max_entry: int = 5,
                 min_exit: int = 1, max_exit: int = 4) -> bool:
        """
        Check if the individual meets the constraints.

        Args:
            min_entry: Minimum number of entry indicators
            max_entry: Maximum number of entry indicators
            min_exit: Minimum number of exit indicators
            max_exit: Maximum number of exit indicators

        Returns:
            True if constraints are satisfied
        """
        n_entry = self.n_entry_selected
        n_exit = self.n_exit_selected

        return (min_entry <= n_entry <= max_entry and
                min_exit <= n_exit <= max_exit)

    def copy(self) -> 'Individual':
        """Create a deep copy of this individual."""
        new_ind = Individual(
            n_indicators=self.n_indicators,
            vector=self.vector.copy(),
            fitness=self.fitness,
            entry_indicators=self.entry_indicators.copy(),
            exit_indicators=self.exit_indicators.copy(),
            iteration=self.iteration,
            index=self.index,
        )
        new_ind.strategy_name = self.strategy_name
        new_ind.backtest_result = self.backtest_result.copy()
        return new_ind

    def __repr__(self) -> str:
        return (f"Individual({self.strategy_name}, "
                f"entry={self.n_entry_selected}, exit={self.n_exit_selected}, "
                f"fitness={self.fitness})")

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "strategy_name": self.strategy_name,
            "iteration": self.iteration,
            "index": self.index,
            "entry_indicators": self.entry_indicators,
            "exit_indicators": self.exit_indicators,
            "n_entry": self.n_entry_selected,
            "n_exit": self.n_exit_selected,
            "fitness": self.fitness,
            "backtest_result": self.backtest_result,
            "vector": self.vector.tolist(),
        }
