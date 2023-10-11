################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os

from evoman.environment import Environment

import numpy as np
# we want an evolutionary algorithm that can learn to play the game
from neat import nn, population, statistics, parallel, DefaultGenome, config, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation, StdOutReporter, StatisticsReporter
# we want to visualize the results
import matplotlib.pyplot as plt
import seaborn as sns
# we want to save the results
import pickle

# imports other libs
import time
import glob
import pygame
import random
import csv

# random.seed(69)
# np.random.seed(69)

config_file = 'config_neat.sh'

experiment_name = 'test'

NAME = "Run_Hidden"

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

def eval_genomes(genomes, config, en=1, verbose=False):
    clock = pygame.time.Clock()

    nets = []
    ge = []
    environments = []

    global fitness_over_generations
    fitness_over_generations = []

    if verbose:
        print("Enemy: ", en)
    for g, gen in genomes:
        net = nn.FeedForwardNetwork.create(gen, config)
        nets.append(net)

        # use the network to play the game
        start = time.time()

        env = Environment(experiment_name=experiment_name,
                        enemymode='static',
                        speed="fastest", # or normal
                        sound="off",
                        fullscreen=False,
                        use_joystick=True,
                        playermode='ai',
                        visuals=False)
        env.update_parameter('enemies', [en])
        env.play()

        gen.fitness = round(env.fitness_single(), 2)
        ge.append(gen)
        
        environments.append(env)
        if verbose:
            print("Finished enemy")
            print()

        fitness_over_generations.append(gen.fitness)

def save_genome(genome, enemy, filename):
    global iteration
    cur_time = time.strftime("%H:%M:%S")
    fitness = round(genome.fitness, 2)
    out_name = f"{filename}_I({iteration})_E({enemy})_FIT({fitness})_T({cur_time}).txt"
    out_file = os.path.join("testLogs/", out_name)
    with open(out_file, 'wb') as f:
        pickle.dump(genome, f)

def run(config_path):
    # Load configuration.
    best_per_enemy = []
    enemies = ENEMIES

    # eval_par = parallel.ParallelEvaluator(2, eval_genomes)

    # for en in enemies[:1]:
    #     best_genome = p.run(eval_par.evaluate, en)
    #     best_per_enemy.append(best_genome)
    #     print("Best fitness -> {}".format(best_genome))
    #     save_genome(best_genome, en, 'Test_111')

    verbose = False

    best_per_gen = []
    if verbose:
        print("Enemy:", en)

    cfg = config.Config(DefaultGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation, config_path)
    p = population.Population(cfg)
    p.add_reporter(StdOutReporter(True))
    stats = StatisticsReporter()
    p.add_reporter(stats)

    global fitness_over_generations
    global fitness_per_enemy
    best_genome = p.run(eval_genomes, en, GENERATIONS)
    print("Enemy:", en)
    print(fitness_over_generations)

    mean_fitness = round(np.mean(fitness_over_generations), 2)
    best_fitness = round(best_genome.fitness, 2)

    save = {"Generations": fitness_over_generations, "Mean": mean_fitness, "Best": best_fitness}
    fitness_per_enemy[en] = save

    best_per_gen.append(best_genome)

    if verbose:
        print("Best fitness -> {}".format(best_genome))

    save_genome(best_genome, en, NAME)
    # time.sleep(3)
    if verbose:
        print("Average fitness PER enemy -> {}".format(np.mean([x.fitness for x in best_per_gen])))

    best_per_gen = max(best_per_gen, key=lambda x: x.fitness)
    best_per_enemy.append(best_per_gen)

    if verbose:
        print("Best fitness PER enemy -> {}".format(best_per_gen))


def plot_results(results):
    # plot the results
    plt.figure(figsize=(15, 10))
    plt.title("Fitness over time")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.plot(results)
    plt.savefig("fitness_over_time.png")
    plt.show()

# Some global variables
ITERATIONS = 10
iteration = 1
GENERATIONS = 30
ENEMIES = [1, 2, 4]
fitness_over_iterations = []
fitness_over_generations = []
fitness_per_enemy = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}

# RUNNING THE GAME
if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config_neat.sh")
    for en in ENEMIES:
        for i in range(ITERATIONS):
            print("Iteration:", i)
            iteration = i
            run(config_path)
            fitness_over_iterations.append(fitness_per_enemy)
    
    # save the results
    # let's do CSV
    

    # get all files in the folder
    files = glob.glob('testLogs/*.txt')

    # get the latest file
    latest_file = max(files, key=os.path.getctime)

    # load the latest file
    with open(latest_file, 'rb') as f:
        bpe = pickle.load(f)
        print(bpe)



# population = population.Population('neat_config')
# statistics.save_stats(population.statistics)

# eval_fitness = 0.9*(100 - self.get_enemylife()) + 0.1*self.get_playerlife() - np.log(self.get_time())

# population.run(eval_fitness, GENERATIONS)
# eval = parallel.ParallelEvaluator(5, eval_fitness)

# # initializes environment with human player and static enemies
# print("Initializing environment")
# for en in range(1, 9):
#     print("Enemy: ", en)
#     # out of each generation, we want to save the best individual
#     best_individual = None
#     best_fitness = 0
#     for gen in range(GENERATIONS):
#         print("Generation: ", gen)
#         start = time.time()
#         env = Environment(experiment_name=experiment_name,
#                         enemymode='static',
#                         speed="fastest", # or normal
#                         sound="off",
#                         fullscreen=False,
#                         use_joystick=True,
#                         playermode='ai',
#                         visuals=False)
#         env.update_parameter('enemies', [en])
#         env.play()

#         fitness = env.fitness_single()

#         # print results
#         print('Results:')
#         print('Time elapsed:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start)))
#         print('Fitness:', fitness)
#         print('\n')

#         # save the best individual
#         if fitness > best_fitness:
#             best_fitness = fitness
#             best_individual = env.player_controller

#         # use neat to evolve the best individual
        
#         nn.save(best_individual, 'best_individual.txt')