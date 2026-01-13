import numpy as np
from crypto_predictor import Predictor


def objective_function(coin: Predictor, candidate, target_variable):
    alpha, beta, gamma, feat_thres = candidate
    coin.alpha = alpha,
    coin.beta = beta
    coin.gamma = gamma
    coin.feature_threshold_coff = feat_thres
    coin.pre_run(target_variable)
    fitness, cols_count, cols, parameters = coin.run(predict_column=target_variable)
    return fitness, cols_count, cols, parameters

def rao3_algorithm(coin:Predictor, target_variable:str, lower_bound:list, upper_bound:list, iterations:int = 50, pop_size: int = 10, obj_func  = objective_function):
    log_list = []

    if len(lower_bound) != len(upper_bound):
        raise Exception("upper and lower bound have not same length")
    else:
        dimension = len(lower_bound)

    best_solution_of_iterations = np.zeros((iterations+1, dimension))
    best_fitness_of_iterations = np.zeros(iterations+1)

    worst_solution_of_iterations = np.zeros((iterations+1, dimension))
    worst_fitness_of_iterations = np.zeros(iterations+1)

    population = np.random.uniform(low=lower_bound, high=upper_bound, size=(pop_size, dimension))
    fitness_pop = np.zeros(pop_size)
    cols_dict = {}
    params_dict = {}

    for pop_index, solution in enumerate(population):
        fitness, cols_count, cols, parameters = obj_func(coin, solution, target_variable)
        fitness_pop[pop_index] = fitness
        cols_dict[pop_index] = cols
        params_dict[pop_index] = parameters
        print(f"iter:-1 indv:{pop_index} fitness:{fitness:,.2f}")

    for iter in range(iterations):
      
        best_fitness_iter = np.min(fitness_pop)
        best_solution_iter = population[np.argmin(fitness_pop),:]
        best_solution_of_iterations[iter,:] = best_solution_iter
        best_fitness_of_iterations[iter] = best_fitness_iter 

        worst_fitness_iter = np.max(fitness_pop)
        worst_solution_iter = population[np.argmax(fitness_pop),:]
        worst_solution_of_iterations[iter,:] = worst_solution_iter
        worst_fitness_of_iterations[iter] = worst_fitness_iter 

        #Update
        new_population = np.ones((pop_size, dimension))
        for pop_index, solution in enumerate(population):
            r = np.random.randint(pop_size)
            while pop_index == r:   
                r = np.random.randint(pop_size)
            random_candidate = population[r,:]
            fitness_solution = fitness_pop[pop_index]
            fitness_candidate = fitness_pop[r]

            temp_good_sols_mean = [sol * f if f <= fitness_solution else np.zeros(dimension) for f, sol in zip(fitness_pop, population)]
            temp_bad_sols_mean = [sol * f if f >= fitness_solution else np.zeros(dimension) for f, sol in zip(fitness_pop, population)]
            

            good_sol = np.sum(temp_good_sols_mean, axis=0) / np.sum(fitness_pop[fitness_pop<=fitness_solution])
            bad_sol = np.sum(temp_bad_sols_mean, axis=0) / np.sum(fitness_pop[fitness_pop>=fitness_solution])

            r1 = np.random.rand() - 0.5
            r2 = np.random.rand() - 0.5
            r3 = np.random.rand() - 0.5

            r_list = [r1, r2, r3]
            r_list = [2 * (r-np.min(r_list)) / (np.max(r_list) - np.min(r_list)) - 1 for r in r_list]
            r1, r2, r3 = r_list

            if fitness_solution > fitness_candidate:
                new_sol = solution + r1 * (best_solution_iter - worst_solution_iter) + r2 * (solution - random_candidate) + r3 * (good_sol - bad_sol)
            else:
                new_sol = solution + r1 * (best_solution_iter - worst_solution_iter) + r2 * (random_candidate - solution) + r3 * (good_sol - bad_sol)
            new_sol = np.clip(new_sol, lower_bound, upper_bound)
            new_population[pop_index] = np.copy(new_sol)

        #ReEvaluation
        for pop_index, solution in enumerate(new_population):
            fitness, cols_count, cols, parameters = obj_func(coin, solution, target_variable)
            if fitness_pop[pop_index] > fitness:
                fitness_pop[pop_index] = fitness
                cols_dict[pop_index] = cols
                params_dict[pop_index] = parameters
                population[pop_index] = np.copy(solution)
            print(f"iter:{iter} indv:{pop_index} fitness:{fitness:,.2f}")

    global_best = np.min(fitness_pop)
    global_best_solution = population[np.argmin(fitness_pop)]
    columns = cols_dict[np.argmin(fitness_pop)]
    params = params_dict[np.argmin(fitness_pop)]
    return global_best, global_best_solution, columns, params
