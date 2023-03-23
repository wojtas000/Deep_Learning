"""
Module for Weighted random search algorithm implementation. Not finished due to problems with downloading fANOVA package.
"""

import random

def wrs_step(F, params:dict, F_value:float, p:dict, k:dict, iteration:int, param_grid:dict):
    thresh = random.uniform(0,1)
    new_params = {}
    for hypar in params:
        if p[hypar]>=thresh or iteration < k[hypar]:
            new_params[hypar] = random.choice(param_grid[hypar])
        else:
            new_params[hypar] = params[hypar]
    if new_params == params:
        return params, F_value
    
    F_new_value = F(new_params)

    if F_new_value > F_value:
        return new_params, F_new_value  
    else:
        return params, F_value
    
def fanova(param_history, F_value_history, param_grid):
    p = {key: 1 for key in param_grid}
    k = {key: 3 for key in param_grid}
    return p, k

def wrs(F, N:int, N_0:int, param_grid:dict):

    starting_params = {key: random.choice(value) for key, value in param_grid.items()}
    param_history = [0 for i in range(N_0)]
    F_value_history = [0 for i in range(N_0)]
    p = {key: 1 for key in param_grid}
    k = {key: 3 for key in param_grid}
    params = starting_params
    F_value = F(params)
    print(params, F_value)
    
    for iter in range(N_0):
        params, F_value = wrs_step(F=F, params=params, F_value=F_value, p=p, k=k, 
                                   iteration=iter, param_grid=param_grid )
        param_history[iter] = params
        F_value_history[iter] = F_value
        print(params, F_value)
    
    p, k = fanova(param_history, F_value_history, param_grid)


    for iter in range(N - N_0):
        params, F_value = wrs_step(F=F, params=params, F_value=F_value, p=p, k=k, 
                                   iteration=iter, param_grid=param_grid )
        print(params, F_value)
    
    return params, F_value    


if __name__ == '__main__':
    
    def add(params):
        sum = 0
        for param in params:
            sum += params[param]
        return sum

    params = {'lr': 0.01, 'batch_size': 32}
    param_grid = {'lr': [0.01, 0.05, 0.1], 'batch_size': [32, 64, 128]}
    p = {'lr': 0.5, 'batch_size': 0.5}
    k = {'lr': 3, 'batch_size': 5}
    iteration = 4
    F = add
    F_value = F(params)
    N, N_0 = 10, 3
    print(wrs_step(F, params, F_value, p, k, iteration, param_grid))
    print(wrs(F, N, N_0, param_grid))