import sys, os
sys.path.insert(0, 'evoman')
sys.path.insert(1, 'extra')
from evoman.environment import Environment
from controllers import generalist
import pickle
import neat
from extra.es_hyperneat import ESNetwork
from extra.hyperneat import create_phenotype_network
from extra.substrate import Substrate
import csv

# Run faster
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Initialize an environment for a specialist game (single objective) with a static enemy and an ai-controlled player

NAME = 'Custom_generalist'
# ENEMIES = [1,2,3,4,5,6,7,8]
ENEMIES = range(1, 9)
OPPONENT = "".join(map(str, ENEMIES))
NEAT, ESNEAT = False, True
ETRAIN = "146"
ITER = 6

if type(ENEMIES) == list:
    MULTI = "yes"
else:
    MULTI = "no"

# if file does not exist write header
if not os.path.isfile(r"test.csv"):
    statsfile = open(r"test.csv", "w")
    w_csv = csv.writer(statsfile)
    w_csv.writerow(["EXPN", "EA", "ENEMY", "Iteration", "Gain"])
    statsfile.close()

# Open boxplot stats file 
statsfile = open(r"test.csv", "a")
w_csv = csv.writer(statsfile)

if __name__ == "__main__":

    for e in ENEMIES:
        env = Environment(experiment_name='logs',
              playermode="ai",
              multiplemode=MULTI,
              enemies=[e],
              player_controller=generalist(),
              speed="fastest",
              enemymode="static",
              level=2,
              visuals=False)
        
        print("Enemy", e)

        for i in range(0, 5):
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

            print("Iteration", i)
        
            a = env.play(nn)
            print(a)

            gain = a[1] - a[2]
            print(gain)

            w_csv.writerow([NAME, f"{ETRAIN}_{ITER}", e, i, gain])

    statsfile.close()