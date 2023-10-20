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

# # Determines whether NEAT or the simple NN is being used
# NEAT = len(sys.argv) == 1
# If more than one argument is given, then we use a modified version of NEAT
# Setting up the arguments with their corresponding values

VERBOSE = True
USE_ARGS = True

EXPERIMENT_NAME="Custom_generalist"
PLAYER_MODE="ai"
ENEMIES=[1]
CONTROLLER=generalist()
MULTI_MODE="no"
SPEED="fastest"
ENEMY_MODE="static"
LEVEL=2
VISUALS=True
GENS=30
ITERATIONS=1
PARALLEL=False
PARALLEL_EVALS=multiprocessing.cpu_count() // 2
NEAT, ESNEAT = False, False

if USE_ARGS and len(sys.argv) > 1:
    EXPERIMENT_NAME = sys.argv[1]
    PLAYER_MODE = sys.argv[2]
    ENEMIES = sys.argv[3]
    if ENEMIES == "all":
        ENEMIES = [1,2,3,4,5,6,7,8]
    else:
        ENEMIES = list(map(int, ENEMIES.split(',')))
    CONTROLLER = sys.argv[4]
    if CONTROLLER == "generalist":
        CONTROLLER = generalist()
    MULTI_MODE = sys.argv[5]
    SPEED = sys.argv[6]
    ENEMY_MODE = sys.argv[7]
    LEVEL = int(sys.argv[8])
    VISUALS = int(sys.argv[9])
    GENS = int(sys.argv[10])
    ITERATIONS = int(sys.argv[11])
    PARALLEL = int(sys.argv[12])

if VERBOSE:
        print("_" * 50)
        print("Experiment name: %s" % EXPERIMENT_NAME)
        print("Player mode: %s" % PLAYER_MODE)
        print("Enemies: %s" % ENEMIES)
        print("Controller: %s" % CONTROLLER)
        print("Multi mode: %s" % MULTI_MODE)
        print("Speed: %s" % SPEED)
        print("Enemy mode: %s" % ENEMY_MODE)
        print("Level: %s" % LEVEL)
        print("Visuals: %s" % (True if VISUALS else False))
        print("Generations: %s" % GENS)
        print("Iterations: %s" % ITERATIONS)
        print("Parallel evaluations: %s" % (PARALLEL_EVALS if PARALLEL else 1))
        print("_" * 50)

# Holds the best genomes for each generation
best_genomes = []
# 
# Determine whether NEAT or ESNEAT is being used
match EXPERIMENT_NAME[0]:
    # Case CUSTOM
    case "C":
        NEAT, ESNEAT = False, False
    # Case NEAT
    case "N":
        NEAT, ESNEAT = True, False
    # Case ESNEAT
    case "E":
        NEAT, ESNEAT = False, True
    case default:
        print("Please specify whether you want to use NEAT, ESNEAT, or a custom algorithm!")


# Make the module headless to run the simulation faster
if not VISUALS:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# Initialize an environment for a generalist game (single objective) with a static enemy and an ai-controlled player
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

if len(ENEMIES) == 1:
    env.update_parameter('enemies', [1])

def run(config):
    global PARALLEL_EVALS
    # Create the population, which is the top-level object for a NEAT run.
    population = neat.Population(config)
    # Add a stdout reporter to show progress in the terminal.
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Run for 10 generations
    # Use multiple evaluations in parallel
    if PARALLEL:
        pe = neat.ParallelEvaluator(PARALLEL_EVALS, eval_genome)
        winner = population.run(pe.evaluate, GENS)
    # Use single-threaded evaluations
    else:
        winner = population.run(evaluate, GENS)
    return winner, stats

def eval_genome(genome, config):
    global NEAT, ESNEAT
    """
    This function will be run in parallel by ParallelEvaluator.  It takes two
    arguments (a single genome and the genome class configuration data) and
    should return one float (that genome's fitness).

    Note that this function needs to be in module scope for multiprocessing.Pool
    (which is what ParallelEvaluator uses) to find it.  Because of this, make
    sure you check for __main__ before executing any code (as we do here in the
    last few lines in the file), otherwise you'll have made a fork bomb
    instead of a neuroevolution demo. :)
    """

    if ESNEAT:
        sub = Substrate(20, 5)
        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        network = ESNetwork(sub, cppn)
        nn = network.create_phenotype_network()
    else:
        nn = neat.nn.FeedForwardNetwork.create(genome, config)

    fitness = env.play(nn)[0]

    # Get the mean fitness of the population
    print(genome)

    return fitness

def evaluate(genomes, config):
    global NEAT, ESNEAT
    best = 0
    if ESNEAT:
        sub = Substrate(20, 5)
    for genome_id, genome in genomes:
        # Create a 10 Hidden Neuron multilayer for each genome
        if NEAT:
            nn = neat.nn.FeedForwardNetwork.create(genome, config)
        elif ESNEAT:
            cppn = neat.nn.FeedForwardNetwork.create(genome, config)
            network = ESNetwork(sub, cppn)
            nn = network.create_phenotype_network()
        else:
            nn = neat.nn.FeedForwardNetwork.create(genome, config)
        # Make each genome (individual) play the game
        f,p,e,t = env.play(nn)
        # Assign a fitness value to a specific genome
        genome.fitness = f
        best = genome.fitness if genome.fitness > best else best
    best_genomes.append(best)

def process_results(winner, stats):
    # Use NEAT's Population object to obtain the statistics you want
    # Create or open a csv file called StatsFile.csv that can be written in from last position 
    filename = r"stats/%s_%s_StatsFile.csv" % (env.enemies, ("neat" if NEAT else "esneat" if ESNEAT else "custom"))

    HEADER = ['Generation', 'Mean', 'Best']
    # if file does not exist, create it and add the header
    if not os.path.isfile(filename):
        with open(filename, "w") as file:
            writer = csv.writer(file)
            writer.writerow(HEADER)

    with open(filename, "a") as file:
        # Create a csv writer object
        writer = csv.writer(file)

        # Get list of means
        mean = stats.get_fitness_mean()
        best_genome = stats.best_genome()
        
        # Add empty values to separate runs (iterations)
        EMPTY = [-1] * len(HEADER)

        # Loop through mean lists to add values to file
        for i in range(len(mean)):
            final_mean = mean[i]
            best_fitness = float(best_genome.fitness)
            writer.writerow([i, final_mean, best_fitness])

        writer.writerow(EMPTY)

def main(params) -> None:
    global EXPERIMENT_NAME, NEAT, ESNEAT
    config = params[0]
    # Run simulations to determine a solution
    winner, stats = run(config)

    enemies = "".join(map(str, ENEMIES))
    winner_path = f"winners/{EXPERIMENT_NAME}_{enemies}_{params[1]}_{'neat' if NEAT else 'esneat' if ESNEAT else 'custom'}-winner.pkl"

    # Save the winner
    with open(winner_path, "wb") as f:
        pickle.dump(winner, f)

    # Process results
    process_results(winner, stats)

cur_iter = 0

if __name__ == "__main__":
    # Create the folder for Assignment 2
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Network parameters
    # num_inputs = 20
    # num_hidden = 10
    # num_outputs = 5

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'configs/' + ('neat_generalist.cfg' if NEAT else 'esneat_generalist.cfg' if ESNEAT else 'custom_generalist.cfg'))

    # # add bias node to input layer
    # # print(help(config.genome_config))

    # # print config
    # print(config.genome_config.input_keys)

    # # get all config keys
    # print(config.genome_config)

    # # # print hidden layer
    # # print(config.genome_config.hidden_keys)

    # # # print output layer
    # # print(config.genome_config.output_keys)

    if ITERATIONS == 1:
        main((config, 0))
    else:
        for i in range(ITERATIONS):
            main((config, i + 1))
        # with ProcessPoolExecutor() as executor:
        #     executor.map(main, [(config, i) for i in range(ITERATIONS)])
    
    
