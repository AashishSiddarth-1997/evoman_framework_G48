################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os

import numpy as np
from evoman.controller import Controller
from evoman.environment import EnvironmentEA
from RollingHorizonEA.rhea import RollingHorizonEvolutionaryAlgorithm
# from RollingHorizonEA.environment import Environment

experiment_name = 'ai'

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Oldies
# num_dims = 600
# m = 50
# num_evals = 50
# rollout_length = 10
# mutation_probability = 0.1

num_dims = 666
m = 69
num_evals = 100
rollout_length = 10
mutation_probability = 0.5

# initializes environment with human player and static enemies
for en in range(1, 9):
    # # Set up the problem domain as m-max game
    # env = EnvironmentEA(experiment_name=experiment_name,
    #                   enemymode='static',
    #                   speed="fastest",
    #                   sound="off",
    #                   fullscreen=False,
    #                   use_joystick=True,
    #                   playermode='ai',
    #                   visuals=False)
    
    # env.update_parameter('enemies', [en])

    # sol = np.loadtxt('solutions_demo/demo_' + str(en) + '.txt')
    # print('\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY ' + str(en) + ' \n')

    # env.play(sol)

    generations = range(100)
    pcont = Controller()

    for gen in generations:
        print("Running generation", gen)

        # Set up the problem domain as m-max game
        env = EnvironmentEA(experiment_name=experiment_name,
                        enemymode='static',
                        speed="fastest",
                        sound="off",
                        fullscreen=False,
                        use_joystick=True,
                        playermode='ai',
                        player_controller=pcont,
                        visuals=False)
        
        env.update_parameter('enemies', [en])

        env.play(pcont=pcont)

        rhea = RollingHorizonEvolutionaryAlgorithm(rollout_length, env, mutation_probability, num_evals)

        rhea.run()


