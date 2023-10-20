import sys, os, multiprocessing
sys.path.insert(0, 'evoman')
sys.path.insert(1, 'extra')
from evoman.environment import Environment
from controllers import generalist
import pickle
import neat
import csv
from concurrent.futures import ProcessPoolExecutor
from extra.substrate import Substrate
from extra.es_hyperneat import ESNetwork
from extra.hyperneat import create_phenotype_network

# os.environ["SDL_VIDEODRIVER"] = "dummy"

EXPERIMENT_NAME="Custom_generalist"
PLAYER_MODE="ai"
ENEMIES=range(1, 9)
CONTROLLER=generalist()
MULTI_MODE="yes"
SPEED="normal"
ENEMY_MODE="static"
LEVEL=2
VISUALS=True

env = Environment(experiment_name=EXPERIMENT_NAME,
              playermode=PLAYER_MODE,
              enemies=ENEMIES,
              player_controller=CONTROLLER,
              multiplemode=MULTI_MODE,
              speed=SPEED,
              enemymode=ENEMY_MODE,
              level=LEVEL,
              visuals=VISUALS,
              logs='off')

NAME = "Custom_generalist"
ETRAIN = "146"
ITER = "6"
NEAT, ESNEAT = False, True

if __name__ == "__main__":

    # Initialize the NEAT config 
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'configs/' + ('custom_generalist.cfg' if NEAT else 'esneat_generalist.cfg'))
    winner_name = f"{NAME}_{ETRAIN}_{'NEAT' if NEAT else 'ESNEAT'}_{ITER}"
    with open(f"winners/{winner_name}_winner.pkl", "rb") as f:
        unpickler = pickle.Unpickler(f)
        genome = unpickler.load()
    # Create either an Feedforward Network or a CPPN
    if NEAT:
        nn = neat.nn.FeedForwardNetwork.create(genome, config)
    else:
        sub = Substrate(20, 5)
        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        network = ESNetwork(sub, cppn)
        nn = network.create_phenotype_network()

    a = env.play(nn)
    print(a)