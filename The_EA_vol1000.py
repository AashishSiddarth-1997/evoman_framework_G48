# imports other libs
import sys
from evoman.environment import Environment
from demo_controller import player_controller
import time
import numpy as np
from math import fabs,sqrt
import os
import random

def fitness(env,x):
    fit, p_life, enemy_life, times = env.play(pcont=x)
    f = 0.9 * (100 - enemy_life) + 0.1 * p_life - np.log(times)
    return f

def evaluate(x):
    return np.array(list(map(lambda y:fitness(env,y),x)))

def roulette_wheel(population, fitness_score):
    total_fitness = sum(fitness_score)
    selection_probabilities = [(fitness / total_fitness) ** 2 for fitness in fitness_score]
    spin = random.uniform(0, 1)
    
    cumulative_probability = 0
    for i, probability in enumerate(selection_probabilities):
        cumulative_probability += probability
        if spin <= cumulative_probability:
            return population[i]
    
    # If no individual was selected, return a default individual
    return random.choice(population)


def simple_arithmetic_crossover(parent1, parent2, crossover_point, crossover_rate=0.5):
    # Determine the length of the parents
    parent_length = len(parent1)

    # Ensure that the crossover point is within the valid range
    crossover_point = min(max(crossover_point, 0), parent_length)

    avg_after = np.asarray([(p1 + p2) / 2 for p1, p2 in zip(parent1[crossover_point:], parent2[crossover_point:])])
    avg_before = np.asarray([(p1 + p2) / 2 for p1, p2 in zip(parent1[:crossover_point], parent2[:crossover_point])])

    if random.random() < crossover_rate:
        offspring1 = np.concatenate((parent1[:crossover_point], avg_after))
    else:
        offspring1 = np.concatenate((parent1[crossover_point:], avg_before))

    if random.random() < crossover_rate:
        offspring2 = np.concatenate((parent2[:crossover_point], avg_after))
    else:
        offspring2 = np.concatenate((parent2[crossover_point:], avg_before))

    return offspring1, offspring2

def crossover(population, fitness_score, mutation_rate):
    N_VARS = len(population[0])
    children = []

    for _ in range(len(population) // 5):
        parent1 = roulette_wheel(population, fitness_score)
        parent2 = roulette_wheel(population, fitness_score)

        low, high = round(N_VARS * 0.1), round(N_VARS * 0.9)
        crossover_point = random.randint(low, high)
        offspring1, offspring2 = simple_arithmetic_crossover(parent1, parent2, crossover_point)

        children.extend([offspring1, offspring2])
    
    children = np.array(children)

    # GAUSSIAN MUTATION
    mutation_mask = np.random.rand(*children.shape) < mutation_rate
    mutation_values = np.random.normal(0, 1, size=children.shape)
    children += mutation_mask * mutation_values

    # Save top 50% of the population
    children_fit = evaluate(children)
    # Sort children by the fitness values in the previous step
    top_children = children[np.argsort(children_fit)][len(children) // 2:]

    return top_children

ENEMY = 4
EXP_NAME = "EA_Roulette_" + str(ENEMY)

if not os.path.exists(EXP_NAME):
    os.makedirs(EXP_NAME)

# CONSTANTS
RUN_MODE ='train'
ENEMY = 4
MUTATION_RATE = 0.33
N_HIDDEN_NEURONS = 10
POP_NUM = 100
GENERATIONS = 30
ITERATIONS = 10

# variables
ini = time.time()
last_best = 0

env = Environment(experiment_name=EXP_NAME,
        enemies=[ENEMY],
        playermode='ai',
        player_controller=player_controller(N_HIDDEN_NEURONS),
        enemymode="static",
        level=2,
        speed='fastest',
        visuals=False)

if __name__ == "__main__":

    for iteration in range(ITERATIONS):
        if RUN_MODE=='test':
            best_sol = np.loadtxt(EXP_NAME+f'/best_{iteration}.txt')
            print('\n Running save with best solution\n')
            env.update_parameter('speed','normal')
            env.update_parameter('visuals',str(True))
            evaluate([best_sol])
            sys.exit(0)
        else:
            headless = True
            if headless:
                os.environ["SDL_VIDEODRIVER"] = "dummy"


        N_VARS = (env.get_num_sensors() + 1) * N_HIDDEN_NEURONS + (N_HIDDEN_NEURONS + 1) * 5
        print( '\nNEW EVOLUTION\n')

        population = np.random.uniform(-1,1,(POP_NUM, N_VARS))
        fit_population = evaluate(population)
        best = np.argmax(fit_population)
        mean = np.mean(fit_population)
        std =np.std(fit_population)
        best_p=population
        Champ=fit_population[best]

        for i in range(GENERATIONS):
            offsping = crossover(population,fit_population,MUTATION_RATE)
            fit_offsping = evaluate(offsping)
            worst_population = np.argsort(fit_population)[:len(offsping)]
            population[worst_population] = offsping
            fit_population[worst_population] = fit_offsping
            
            best=np.argmax(fit_population)
            mean = np.mean(fit_population)
            std =np.std(fit_population)
            fit_population[best]=float(evaluate(np.array([population[best]]))[0])
            best_sol = fit_population[best]

            # saves results
            file_aux  = open(EXP_NAME+'/results.txt','a')
            print( '\n GENERATION '+str(i)+' '+str(round(fit_population[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
            file_aux.write('\n'+str(i)+' '+str(round(fit_population[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
            file_aux.close()

            # saves generation number
            file_aux  = open(EXP_NAME+'/gen.txt','w')
            file_aux.write(str(i))
            file_aux.close()

            # saves file with the best solution
            np.savetxt(EXP_NAME+f'/best_{iteration}.txt', population[best])   

            if Champ <= fit_population[best] :
                Champ=fit_population[best]
                best_p=population
                champ_fit=evaluate(best_p)
                solutions=[best_p,champ_fit]
                env.update_solutions(solutions)
                env.save_state()

        fim = time.time() # prints total execution time for experiment
        print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
        print( '\nExecution time: '+str(round((fim-ini)))+' seconds \n')
