"""
Artificial Protozoa Optimizer (APO) - Corrected Implementation
==============================================================

A novel bio-inspired metaheuristic algorithm for engineering optimization.

Based on: Wang, X., et al. (2024). "Artificial Protozoa Optimizer (APO): 
A novel bio-inspired metaheuristic algorithm for engineering optimization"
Knowledge-Based Systems, 111737.
DOI: https://doi.org/10.1016/j.knosys.2024.111737

This implementation is verified against the official MATLAB code from:
https://ww2.mathworks.cn/matlabcentral/fileexchange/162656-artificial-protozoa-optimizer

Key corrections from paper-based implementation:
1. Heterotrophic Xnear uses single scalar Flag (±1), not element-wise
2. Proper handling of sorted population and fitness arrays
3. Exact matching of MATLAB indexing logic (converted to 0-based)
"""

import numpy as np
from typing import Callable, List, Optional, Union
from dataclasses import dataclass


@dataclass
class APOResult:
    """Container for APO optimization results."""
    best_position: np.ndarray
    best_fitness: float
    convergence_curve: List[float]
    final_population: np.ndarray
    final_fitness: np.ndarray
    function_evaluations: int
    iterations: int


class ArtificialProtozoaOptimizer:
    """
    Artificial Protozoa Optimizer (APO) - Verified against official MATLAB implementation.
    
    The APO algorithm is inspired by the survival mechanisms of protozoa (euglena),
    modeling their foraging, dormancy, and reproductive behaviors for optimization.
    
    Key behaviors:
    1. Autotrophic Foraging (Exploration): Simulates photosynthesis-driven movement
    2. Heterotrophic Foraging (Exploitation): Simulates nutrient absorption behavior
    3. Dormancy (Exploration): Random regeneration under stress
    4. Reproduction (Exploitation): Binary fission with perturbation
    
    Parameters
    ----------
    objective_func : Callable
        The objective function to minimize. Should accept a 1D numpy array and return a scalar.
    lb : np.ndarray or float
        Lower bounds of the search space.
    ub : np.ndarray or float
        Upper bounds of the search space.
    dim : int
        Number of dimensions (decision variables).
    pop_size : int, optional
        Population size (default: 100, as in MATLAB).
    max_iter : int, optional
        Maximum number of iterations (default: 500).
    np_neighbors : int, optional
        Number of neighbor pairs for foraging (default: 1).
    pf_max : float, optional
        Maximum proportion fraction for dormancy/reproduction (default: 0.1).
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, optional
        Print progress during optimization (default: False).
    
    Examples
    --------
    >>> import numpy as np
    >>> def sphere(x):
    ...     return np.sum(x**2)
    >>> 
    >>> apo = ArtificialProtozoaOptimizer(
    ...     objective_func=sphere,
    ...     lb=-100, ub=100, dim=30,
    ...     pop_size=100, max_iter=500
    ... )
    >>> result = apo.optimize()
    >>> print(f"Best fitness: {result.best_fitness}")
    """
    
    def __init__(
        self,
        objective_func: Callable[[np.ndarray], float],
        lb: Union[np.ndarray, float],
        ub: Union[np.ndarray, float],
        dim: int,
        pop_size: int = 100,
        max_iter: int = 500,
        np_neighbors: int = 1,
        pf_max: float = 0.1,
        seed: Optional[int] = None,
        verbose: bool = False
    ):
        self.objective_func = objective_func
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.np_neighbors = np_neighbors  # np in MATLAB
        self.pf_max = pf_max
        self.verbose = verbose
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Handle bounds - convert scalar to array
        if np.isscalar(lb):
            self.lb = np.full(dim, float(lb))
        else:
            self.lb = np.asarray(lb, dtype=float)
            
        if np.isscalar(ub):
            self.ub = np.full(dim, float(ub))
        else:
            self.ub = np.asarray(ub, dtype=float)
        
        # Validate bounds
        if len(self.lb) != dim or len(self.ub) != dim:
            raise ValueError(f"Bounds must have length {dim}")
        
        # These will be initialized in optimize()
        self.protozoa = None          # Population matrix (ps x dim)
        self.protozoa_fit = None      # Fitness array (ps,)
        self.best_protozoa = None     # Global best position
        self.best_fit = np.inf        # Global best fitness
        self.convergence = []
        self.func_evals = 0
    
    def _evaluate(self, individual: np.ndarray) -> float:
        """Evaluate an individual and increment the counter."""
        self.func_evals += 1
        return self.objective_func(individual)
    
    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Evaluate entire population."""
        return np.array([self._evaluate(ind) for ind in population])
    
    def optimize(self) -> APOResult:
        """
        Run the APO optimization algorithm.
        
        This implementation follows the exact logic of the official MATLAB code.
        
        Returns
        -------
        APOResult
            Object containing optimization results.
        """
        ps = self.pop_size
        dim = self.dim
        np_pairs = self.np_neighbors  # renamed to avoid confusion with numpy
        iter_max = self.max_iter
        Xmin = self.lb
        Xmax = self.ub
        
        # Initialize population: protozoa(i,:) = Xmin + rand(1,dim).*(Xmax-Xmin)
        self.protozoa = Xmin + np.random.random((ps, dim)) * (Xmax - Xmin)
        
        # Evaluate fitness
        self.protozoa_fit = self._evaluate_population(self.protozoa)
        
        # Find initial best
        best_idx = np.argmin(self.protozoa_fit)
        self.best_protozoa = self.protozoa[best_idx].copy()
        self.best_fit = self.protozoa_fit[best_idx]
        self.convergence = [self.best_fit]
        
        # Pre-allocate arrays
        new_protozoa = np.zeros((ps, dim))
        epn = np.zeros((np_pairs, dim))  # effect of paired neighbors
        
        if self.verbose:
            print(f"APO Optimization Started (verified MATLAB implementation)")
            print(f"{'Iter':>6} | {'Best Fitness':>15} | {'Mean Fitness':>15}")
            print("-" * 45)
        
        # Main loop - MATLAB starts at iter=2
        for iteration in range(1, iter_max):
            # Sort population by fitness (ascending)
            # MATLAB: [protozoa_Fit,index] = sort(protozoa_Fit);
            #         protozoa = protozoa(index,:);
            sort_indices = np.argsort(self.protozoa_fit)
            self.protozoa_fit = self.protozoa_fit[sort_indices]
            self.protozoa = self.protozoa[sort_indices]
            
            # Proportion fraction: pf = pf_max * rand
            pf = self.pf_max * np.random.random()
            
            # Rank indices for dormancy/reproduction: ri = randperm(ps, ceil(ps*pf))
            num_dr = int(np.ceil(ps * pf))
            if num_dr > 0:
                ri = np.random.permutation(ps)[:num_dr]  # 0-based indices
            else:
                ri = np.array([], dtype=int)
            
            # Process each protozoan
            for i in range(ps):  # i is 0-based (MATLAB is 1-based)
                i_matlab = i + 1  # Convert to 1-based for formulas
                
                if i in ri:
                    # Dormancy or Reproduction form
                    # pdr = 1/2 * (1 + cos((1 - i/ps) * pi))
                    pdr = 0.5 * (1 + np.cos((1 - i_matlab / ps) * np.pi))
                    
                    if np.random.random() < pdr:
                        # Dormancy: newprotozoa(i,:) = Xmin + rand(1,dim).*(Xmax-Xmin)
                        new_protozoa[i] = Xmin + np.random.random(dim) * (Xmax - Xmin)
                    else:
                        # Reproduction
                        # Flag = [1,-1](ceil(2*rand)) -> random +1 or -1
                        Flag = 1 if np.random.random() < 0.5 else -1
                        
                        # Mr = zeros, then Mr(randperm(dim, ceil(rand*dim))) = 1
                        Mr = np.zeros(dim)
                        num_mr = int(np.ceil(np.random.random() * dim))
                        if num_mr > 0:
                            Mr[np.random.permutation(dim)[:num_mr]] = 1
                        
                        # newprotozoa(i,:) = protozoa(i,:) + Flag*rand*(Xmin+rand(1,dim).*(Xmax-Xmin)).*Mr
                        rand_scalar = np.random.random()
                        rand_vec = np.random.random(dim)
                        new_protozoa[i] = (self.protozoa[i] + 
                                          Flag * rand_scalar * (Xmin + rand_vec * (Xmax - Xmin)) * Mr)
                else:
                    # Foraging form
                    # f = rand * (1 + cos(iter/iter_max * pi))
                    f = np.random.random() * (1 + np.cos(iteration / iter_max * np.pi))
                    
                    # Mf = zeros, then Mf(randperm(dim, ceil(dim*i/ps))) = 1
                    Mf = np.zeros(dim)
                    num_mf = int(np.ceil(dim * i_matlab / ps))
                    Mf[np.random.permutation(dim)[:num_mf]] = 1
                    
                    # pah = 1/2 * (1 + cos(iter/iter_max * pi))
                    pah = 0.5 * (1 + np.cos(iteration / iter_max * np.pi))
                    
                    if np.random.random() < pah:
                        # Autotrophic form
                        # j = randperm(ps, 1) -> random index (1-based in MATLAB)
                        j = np.random.randint(0, ps)  # 0-based
                        
                        # Calculate neighbor effect
                        epn.fill(0)
                        for k in range(1, np_pairs + 1):
                            # MATLAB 1-based logic converted to 0-based:
                            # if i==1: km=i, kp=i+randperm(ps-i,1)
                            # if i==ps: km=randperm(ps-1,1), kp=i
                            # else: km=randperm(i-1,1), kp=i+randperm(ps-i,1)
                            
                            if i == 0:  # i_matlab == 1
                                km = 0
                                kp = 1 + np.random.randint(0, ps - 1)  # random from 1 to ps-1
                            elif i == ps - 1:  # i_matlab == ps
                                km = np.random.randint(0, ps - 1)  # random from 0 to ps-2
                                kp = ps - 1
                            else:
                                km = np.random.randint(0, i)  # random from 0 to i-1
                                kp = i + 1 + np.random.randint(0, ps - i - 1)  # random from i+1 to ps-1
                            
                            # wa = exp(-abs(protozoa_Fit(km) / (protozoa_Fit(kp) + eps)))
                            wa = np.exp(-np.abs(self.protozoa_fit[km] / 
                                               (self.protozoa_fit[kp] + np.finfo(float).eps)))
                            
                            # epn(k,:) = wa * (protozoa(km,:) - protozoa(kp,:))
                            epn[k-1] = wa * (self.protozoa[km] - self.protozoa[kp])
                        
                        # newprotozoa(i,:) = protozoa(i,:) + f*(protozoa(j,:)-protozoa(i,:)+1/np*sum(epn,1)).*Mf
                        neighbor_sum = np.sum(epn, axis=0) / np_pairs
                        new_protozoa[i] = (self.protozoa[i] + 
                                          f * (self.protozoa[j] - self.protozoa[i] + neighbor_sum) * Mf)
                    
                    else:
                        # Heterotrophic form
                        epn.fill(0)
                        for k in range(1, np_pairs + 1):
                            # MATLAB logic:
                            # if i==1: imk=i, ipk=i+k
                            # if i==ps: imk=ps-k, ipk=i
                            # else: imk=i-k, ipk=i+k
                            # then clamp to [1, ps]
                            
                            if i == 0:  # i_matlab == 1
                                imk = 0
                                ipk = k  # i_matlab + k = 1 + k, then -1 for 0-based
                            elif i == ps - 1:  # i_matlab == ps
                                imk = ps - 1 - k
                                ipk = ps - 1
                            else:
                                imk = i - k
                                ipk = i + k
                            
                            # Clamp to valid range [0, ps-1]
                            imk = max(0, min(imk, ps - 1))
                            ipk = max(0, min(ipk, ps - 1))
                            
                            # wh = exp(-abs(protozoa_Fit(imk) / (protozoa_Fit(ipk) + eps)))
                            wh = np.exp(-np.abs(self.protozoa_fit[imk] / 
                                               (self.protozoa_fit[ipk] + np.finfo(float).eps)))
                            
                            # epn(k,:) = wh * (protozoa(imk,:) - protozoa(ipk,:))
                            epn[k-1] = wh * (self.protozoa[imk] - self.protozoa[ipk])
                        
                        # Flag = [1,-1](ceil(2*rand)) -> scalar +1 or -1
                        Flag = 1 if np.random.random() < 0.5 else -1
                        
                        # Xnear = (1 + Flag*rand(1,dim)*(1-iter/iter_max)) .* protozoa(i,:)
                        # NOTE: Flag is a SCALAR, rand(1,dim) is a vector
                        rand_vec = np.random.random(dim)
                        Xnear = (1 + Flag * rand_vec * (1 - iteration / iter_max)) * self.protozoa[i]
                        
                        # newprotozoa(i,:) = protozoa(i,:) + f*(Xnear-protozoa(i,:)+1/np*sum(epn,1)).*Mf
                        neighbor_sum = np.sum(epn, axis=0) / np_pairs
                        new_protozoa[i] = (self.protozoa[i] + 
                                          f * (Xnear - self.protozoa[i] + neighbor_sum) * Mf)
            
            # Boundary check
            # MATLAB: newprotozoa = ((newprotozoa>=Xmin)&(newprotozoa<=Xmax)).*newprotozoa
            #                      +(newprotozoa<Xmin).*Xmin+(newprotozoa>Xmax).*Xmax
            new_protozoa = np.clip(new_protozoa, Xmin, Xmax)
            
            # Evaluate new population
            new_protozoa_fit = self._evaluate_population(new_protozoa)
            
            # Greedy selection
            # MATLAB: bin = (protozoa_Fit > newprotozoa_Fit)'
            #         protozoa(bin==1,:) = newprotozoa(bin==1,:)
            improved = new_protozoa_fit < self.protozoa_fit
            self.protozoa[improved] = new_protozoa[improved]
            self.protozoa_fit[improved] = new_protozoa_fit[improved]
            
            # Update global best
            best_idx = np.argmin(self.protozoa_fit)
            self.best_fit = self.protozoa_fit[best_idx]
            self.best_protozoa = self.protozoa[best_idx].copy()
            
            self.convergence.append(self.best_fit)
            
            if self.verbose and (iteration % max(1, iter_max // 20) == 0):
                print(f"{iteration:>6} | {self.best_fit:>15.8e} | {np.mean(self.protozoa_fit):>15.8e}")
        
        if self.verbose:
            print("-" * 45)
            print(f"Optimization Complete!")
            print(f"Best Fitness: {self.best_fit:.8e}")
            print(f"Function Evaluations: {self.func_evals}")
        
        return APOResult(
            best_position=self.best_protozoa.copy(),
            best_fitness=self.best_fit,
            convergence_curve=self.convergence,
            final_population=self.protozoa.copy(),
            final_fitness=self.protozoa_fit.copy(),
            function_evaluations=self.func_evals,
            iterations=iter_max
        )

