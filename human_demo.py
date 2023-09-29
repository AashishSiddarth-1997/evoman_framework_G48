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

# random.seed(69)
# np.random.seed(69)

config_file = 'config_neat.sh'

experiment_name = 'test'

NAME = "Run_Hidden_0rate"

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

def eval_genomes(genomes, config, en=1, verbose=False):
    clock = pygame.time.Clock()

    nets = []
    ge = []
    environments = []

    if verbose:
        print("Enemy: ", en)
    for g, gen in genomes:
        if verbose:
            print("Generation: ", g)
        net = nn.FeedForwardNetwork.create(gen, config)
        nets.append(net)

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

        gen.fitness = env.fitness_single()
        ge.append(gen)
        
        environments.append(env)
        if verbose:
            print("Finished enemy")
            print()

def save_genome(genome, enemy, filename):
    cur_time = time.strftime("%H:%M:%S")
    fitness = round(genome.fitness, 2)
    out_name = f"{filename}_E({enemy})_FIT({fitness})_T({cur_time}).txt"
    out_file = os.path.join("testLogs/", out_name)
    with open(out_file, 'wb') as f:
        pickle.dump(genome, f)

def run(config_path):
    # Load configuration.
    best_per_enemy = []
    enemies = range(1, 9)
    # eval_par = parallel.ParallelEvaluator(2, eval_genomes)

    # for en in enemies[:1]:
    #     winner = p.run(eval_par.evaluate, en)
    #     best_per_enemy.append(winner)
    #     print("Best fitness -> {}".format(winner))
    #     save_genome(winner, en, 'Test_111')

    verbose = False

    for en in enemies:
        best_per_gen = []
        if verbose:
            print("Enemy:", en)

        cfg = config.Config(DefaultGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation, config_path)
        p = population.Population(cfg)
        p.add_reporter(StdOutReporter(True))
        stats = StatisticsReporter()
        p.add_reporter(stats)

        winner = p.run(eval_genomes, en, 30)
        # best_genome = winner
        best_per_gen.append(winner)

        if verbose:
            print("Best fitness -> {}".format(winner))

        save_genome(winner, en, NAME)
        # time.sleep(3)
        if verbose:
            print("Average fitness PER enemy -> {}".format(np.mean([x.fitness for x in best_per_gen])))

        best_per_gen = max(best_per_gen, key=lambda x: x.fitness)
        best_per_enemy.append(best_per_gen)

        if verbose:
            print("Best fitness PER enemy -> {}".format(best_per_gen))
    
    print("XXX")

# RUNNING THE GAME
if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config_neat.sh")
    run(config_path)

    # get all files in the folder
    files = glob.glob('testLogs/*')

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