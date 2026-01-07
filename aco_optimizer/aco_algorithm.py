"""
Ant Colony Optimization Algorithm for Trading Strategy Indicator Selection.
"""
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from .config import ACO_CONFIG, INDICATOR_HEURISTICS, USER_DATA_DIR
from .individual import Individual
from .strategy_generator import StrategyGenerator
from .evaluator import Evaluator

logger = logging.getLogger(__name__)


class ACOAlgorithm:
    """
    Ant Colony Optimization for selecting optimal indicator combinations
    for trading strategies.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or ACO_CONFIG

        # Initialize strategy generator
        self.generator = StrategyGenerator()
        self.n_indicators = self.generator.n_indicators
        self.indicator_names = self.generator.indicator_names

        # Initialize evaluator
        self.evaluator = Evaluator(self.generator, self.config)

        # Pheromone matrix: [2 * n_indicators] for entry and exit
        # Each position has pheromone for selecting (1) or not selecting (0)
        self.n_dimensions = self.n_indicators * 2
        self.pheromone = np.full(
            (self.n_dimensions, 2),  # [dimension][0=not_select, 1=select]
            self.config["tau_initial"]
        )

        # Heuristic values
        self.heuristic = self._initialize_heuristic()

        # Best solution tracking
        self.best_individual: Optional[Individual] = None
        self.best_fitness = float('-inf')
        self.fitness_history: List[float] = []
        self.iteration_stats: List[Dict] = []

        # Current state
        self.current_iteration = 0

    def _initialize_heuristic(self) -> np.ndarray:
        """
        Initialize heuristic values based on indicator categories.

        Returns:
            Array of heuristic values for each dimension
        """
        heuristic = np.ones((self.n_dimensions, 2))

        for i, name in enumerate(self.indicator_names):
            category = self.generator.get_indicator_category(name)
            weight = INDICATOR_HEURISTICS.get(category, 1.0)

            # Entry indicators (first half)
            heuristic[i, 1] = weight  # Selecting
            heuristic[i, 0] = 1.0     # Not selecting

            # Exit indicators (second half)
            heuristic[self.n_indicators + i, 1] = weight
            heuristic[self.n_indicators + i, 0] = 1.0

        return heuristic

    def _construct_solution(self, iteration: int, ant_index: int) -> Individual:
        """
        Construct a solution (ant path) using pheromone and heuristic values.
        Uses weighted random sampling to avoid bias toward early indicators.

        Args:
            iteration: Current iteration number
            ant_index: Index of the ant

        Returns:
            Constructed individual
        """
        individual = Individual(
            n_indicators=self.n_indicators,
            iteration=iteration,
            index=ant_index
        )

        # Parameters
        alpha = self.config["alpha"]
        beta = self.config["beta"]
        min_entry = self.config["min_entry_indicators"]
        max_entry = self.config["max_entry_indicators"]
        min_exit = self.config["min_exit_indicators"]
        max_exit = self.config["max_exit_indicators"]

        # Select entry indicators using weighted random sampling
        entry_indices = self._weighted_sample_indicators(
            start_dim=0,
            n_indicators=self.n_indicators,
            min_select=min_entry,
            max_select=max_entry,
            alpha=alpha,
            beta=beta
        )
        for idx in entry_indices:
            individual.vector[idx] = 1

        # Select exit indicators using weighted random sampling
        exit_indices = self._weighted_sample_indicators(
            start_dim=self.n_indicators,
            n_indicators=self.n_indicators,
            min_select=min_exit,
            max_select=max_exit,
            alpha=alpha,
            beta=beta
        )
        for idx in exit_indices:
            individual.vector[idx] = 1

        # Decode to indicator names
        individual.decode(self.indicator_names)

        return individual

    def _weighted_sample_indicators(self, start_dim: int, n_indicators: int,
                                     min_select: int, max_select: int,
                                     alpha: float, beta: float) -> List[int]:
        """
        Sample indicators using ACO probability weights.
        Avoids sequential bias by computing all probabilities first.

        Args:
            start_dim: Starting dimension in pheromone matrix (0 for entry, n_indicators for exit)
            n_indicators: Total number of indicators
            min_select: Minimum indicators to select
            max_select: Maximum indicators to select
            alpha: Pheromone importance
            beta: Heuristic importance

        Returns:
            List of selected indicator indices (absolute positions in vector)
        """
        # Calculate selection weights for all indicators
        weights = np.zeros(n_indicators)
        for i in range(n_indicators):
            dim = start_dim + i
            tau_select = self.pheromone[dim, 1]
            eta_select = self.heuristic[dim, 1]
            weights[i] = (tau_select ** alpha) * (eta_select ** beta)

        # Normalize to probabilities
        total_weight = np.sum(weights)
        if total_weight > 0:
            probs = weights / total_weight
        else:
            probs = np.ones(n_indicators) / n_indicators

        # Determine how many to select (random between min and max)
        n_select = np.random.randint(min_select, max_select + 1)

        # Weighted random sampling without replacement
        selected_local = np.random.choice(
            n_indicators,
            size=min(n_select, n_indicators),
            replace=False,
            p=probs
        )

        # Convert to absolute indices
        return [start_dim + i for i in selected_local]

    def _calculate_probability(self, dimension: int, alpha: float, beta: float) -> float:
        """
        Calculate probability of selecting a dimension.

        Args:
            dimension: Index in the solution vector
            alpha: Pheromone importance
            beta: Heuristic importance

        Returns:
            Probability of selection
        """
        tau_select = self.pheromone[dimension, 1]
        tau_not_select = self.pheromone[dimension, 0]
        eta_select = self.heuristic[dimension, 1]
        eta_not_select = self.heuristic[dimension, 0]

        # ACO probability formula
        numerator = (tau_select ** alpha) * (eta_select ** beta)
        denominator = numerator + (tau_not_select ** alpha) * (eta_not_select ** beta)

        return numerator / denominator if denominator > 0 else 0.5

    def _update_pheromones(self, population: List[Individual]):
        """
        Update pheromone matrix based on ant solutions.

        Args:
            population: List of evaluated individuals
        """
        rho = self.config["rho"]
        Q = self.config["Q"]
        tau_min = self.config["tau_min"]
        tau_max = self.config["tau_max"]

        # Evaporation
        self.pheromone *= (1 - rho)

        # Negative reinforcement: reduce pheromones for losing strategies
        # Skip penalty score (-1000) - that's Claude failure, not indicator quality
        penalty_score = self.config["penalty_score"]
        for individual in population:
            if individual.fitness is not None and individual.fitness < 0 and individual.fitness > penalty_score:
                # Scale: Q=10, fitness=-200 → penalty = 0.4 per indicator
                penalty = Q * abs(individual.fitness) / 30000
                for i in range(self.n_dimensions):
                    if individual.vector[i] == 1:
                        self.pheromone[i, 1] -= penalty

        # Deposit pheromones only for profitable strategies (fitness > 0)
        for individual in population:
            if individual.fitness is None or individual.fitness <= 0:
                continue  # Skip zero-trade, losing, and failed strategies

            # Exponential scaling relative to best fitness
            # Scale: Q=10, best solution → delta_tau = 2.0
            if self.best_fitness > 0:
                delta_tau = Q * (individual.fitness / self.best_fitness) ** 2 / 7
            else:
                delta_tau = Q * 0.01  # Fallback if no best yet

            # Only reinforce selected indicators for winners
            for i in range(self.n_dimensions):
                if individual.vector[i] == 1:
                    self.pheromone[i, 1] += delta_tau
                # Don't reinforce "not select" path

        # Clamp pheromone values
        self.pheromone = np.clip(self.pheromone, tau_min, tau_max)

        # # Elite reinforcement: extra deposit for best solution
        # # Scale: Q=10, best=36.93 → elite_delta = 0.74
        # if self.best_individual is not None and self.best_fitness > 0:
        #     elite_delta = Q * self.best_fitness / 5000
        #     for i in range(self.n_dimensions):
        #         if self.best_individual.vector[i] == 1:
        #             self.pheromone[i, 1] += elite_delta

        #     self.pheromone = np.clip(self.pheromone, tau_min, tau_max)

    def _log_pheromone_ranking(self, top_n: int = 5):
        """
        Log top and bottom pheromone values with indicator names.

        Args:
            top_n: Number of top/bottom indicators to show
        """
        # Get pheromone values for "select" (column 1)
        select_pheromones = self.pheromone[:, 1]

        # Build list of (index, pheromone, type, name)
        pheromone_list = []
        for i in range(self.n_dimensions):
            if i < self.n_indicators:
                ind_type = "entry"
                ind_name = self.indicator_names[i]
            else:
                ind_type = "exit"
                ind_name = self.indicator_names[i - self.n_indicators]

            pheromone_list.append({
                "index": i,
                "pheromone": select_pheromones[i],
                "type": ind_type,
                "name": ind_name
            })

        # Sort by pheromone value
        sorted_list = sorted(pheromone_list, key=lambda x: x["pheromone"], reverse=True)

        # Log top N
        logger.info(f"  Top {top_n} pheromones:")
        for item in sorted_list[:top_n]:
            logger.info(f"    [{item['type']:5}] {item['name']}: {item['pheromone']:.3f}")

        # Log bottom N
        logger.info(f"  Bottom {top_n} pheromones:")
        for item in sorted_list[-top_n:]:
            logger.info(f"    [{item['type']:5}] {item['name']}: {item['pheromone']:.3f}")

    def run(self, n_iterations: int = None, n_ants: int = None) -> Individual:
        """
        Run the ACO algorithm.

        Args:
            n_iterations: Number of iterations (default from config)
            n_ants: Population size (default from config)

        Returns:
            Best individual found
        """
        n_iterations = n_iterations or self.config["n_iterations"]
        n_ants = n_ants or self.config["n_ants"]

        logger.info(f"Starting ACO optimization: {n_iterations} iterations, {n_ants} ants")
        logger.info(f"Total indicators: {self.n_indicators}")

        start_time = datetime.now()

        for iteration in range(1, n_iterations + 1):
            self.current_iteration = iteration
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iteration}/{n_iterations}")
            logger.info(f"{'='*60}")

            # Phase 1: Construct all individuals first
            population: List[Individual] = []
            for ant_idx in range(n_ants):
                individual = self._construct_solution(iteration, ant_idx)
                population.append(individual)

            logger.info(f"Constructed {n_ants} individuals, starting parallel evaluation...")

            # Phase 2: Evaluate all individuals in parallel using thread pool
            max_workers = self.config.get("max_parallel_evaluations", 5)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all evaluation tasks
                future_to_individual = {
                    executor.submit(self.evaluator.evaluate, ind): ind
                    for ind in population
                }

                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_individual):
                    individual = future_to_individual[future]
                    try:
                        fitness = future.result()
                        completed += 1
                        if completed % 5 == 0 or completed == n_ants:
                            logger.info(f"  Evaluated {completed}/{n_ants} individuals")
                    except Exception as e:
                        logger.error(f"Evaluation failed for {individual.strategy_name}: {e}")
                        individual.fitness = self.config["penalty_score"]

            # Phase 3: Find best individuals after all evaluations complete
            iteration_best_fitness = float('-inf')
            iteration_best: Optional[Individual] = None

            for individual in population:
                fitness = individual.fitness if individual.fitness is not None else self.config["penalty_score"]

                # Track iteration best
                if fitness > iteration_best_fitness:
                    iteration_best_fitness = fitness
                    iteration_best = individual.copy()

                # Track global best
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_individual = individual.copy()
                    logger.info(f"*** New global best: {fitness:.2f} ***")

            # Update pheromones
            self._update_pheromones(population)

            # Log top/bottom pheromone values
            self._log_pheromone_ranking()

            # Record statistics
            fitnesses = [ind.fitness for ind in population if ind.fitness is not None]
            valid_fitnesses = [f for f in fitnesses if f > self.config["penalty_score"]]

            stats = {
                "iteration": iteration,
                "best_fitness": self.best_fitness,
                "iteration_best": iteration_best_fitness,
                "mean_fitness": np.mean(valid_fitnesses) if valid_fitnesses else 0,
                "std_fitness": np.std(valid_fitnesses) if valid_fitnesses else 0,
                "valid_solutions": len(valid_fitnesses),
                "total_solutions": n_ants,
            }
            self.iteration_stats.append(stats)
            self.fitness_history.append(self.best_fitness)

            # Log progress
            if iteration % self.config["log_interval"] == 0:
                logger.info(f"Iteration {iteration} summary:")
                logger.info(f"  Global best: {self.best_fitness:.2f}")
                logger.info(f"  Iteration best: {iteration_best_fitness:.2f}")
                logger.info(f"  Valid solutions: {len(valid_fitnesses)}/{n_ants}")
                if valid_fitnesses:
                    logger.info(f"  Mean fitness: {np.mean(valid_fitnesses):.2f}")

            # Save checkpoint
            if iteration % self.config["save_best_every"] == 0:
                self._save_checkpoint(iteration)

        # Final summary
        elapsed = datetime.now() - start_time
        logger.info(f"\n{'='*60}")
        logger.info("ACO Optimization Complete")
        logger.info(f"{'='*60}")
        logger.info(f"Total time: {elapsed}")
        logger.info(f"Best fitness: {self.best_fitness:.2f}")
        if self.best_individual:
            logger.info(f"Best strategy: {self.best_individual.strategy_name}")
            logger.info(f"Entry indicators: {self.best_individual.entry_indicators}")
            logger.info(f"Exit indicators: {self.best_individual.exit_indicators}")

        # Save final results
        self._save_results()

        return self.best_individual

    def _save_checkpoint(self, iteration: int):
        """Save checkpoint of current state."""
        checkpoint_dir = USER_DATA_DIR / "aco_checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            "iteration": iteration,
            "best_fitness": self.best_fitness,
            "best_individual": self.best_individual.to_dict() if self.best_individual else None,
            "fitness_history": self.fitness_history,
            "pheromone_stats": {
                "mean": float(np.mean(self.pheromone)),
                "std": float(np.std(self.pheromone)),
                "max": float(np.max(self.pheromone)),
                "min": float(np.min(self.pheromone)),
            },
            "evaluator_stats": self.evaluator.get_stats(),
        }

        path = checkpoint_dir / f"checkpoint_iter_{iteration}.json"
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"Checkpoint saved: {path}")

    def _save_results(self):
        """Save final optimization results."""
        results_dir = USER_DATA_DIR / "aco_results"
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results = {
            "timestamp": timestamp,
            "config": {k: v for k, v in self.config.items() if not callable(v)},
            "best_fitness": self.best_fitness,
            "best_individual": self.best_individual.to_dict() if self.best_individual else None,
            "fitness_history": self.fitness_history,
            "iteration_stats": self.iteration_stats,
            "evaluator_stats": self.evaluator.get_stats(),
            "n_indicators": self.n_indicators,
            "indicator_names": self.indicator_names,
        }

        # Save JSON results
        path = results_dir / f"aco_results_{timestamp}.json"
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved: {path}")

        # Save pheromone matrix
        pheromone_path = results_dir / f"pheromone_{timestamp}.npy"
        np.save(pheromone_path, self.pheromone)

        return path
