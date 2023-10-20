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

def evaluate(genomes, config):
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

def run(config):
    global PARALLEL_EVALS
    # Create the population, which is the top-level object for a NEAT run.
    population = neat.Population(config)
    # Add a stdout reporter to show progress in the terminal.
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(evaluate, GENS)

    return winner, stats

def save_data(stats, iteration):
    # Create a new csv file for each iteration
    filename = f"stats/{EXPERIMENT_NAME}_"

def main(params):
    # best_genomes = []
    config = params[0]
    ITER = params[1]
    # Run simulations to determine a solution
    winner, stats = run(config)

    print("*" * 50)
    print(stats)
    print("Mean fitness: ")
    print(stats.get_fitness_mean())
    print("Median fitness: ")
    print(stats.get_fitness_median())
    print("Stdev fitness: ")
    print(stats.get_fitness_stdev())
    print("Best fitness: ")
    print(stats.best_genome().fitness)
    print("Best fitness over time: ")
    print(best_genomes)
    print("*" * 50)

    winner_path = f"winners/{OUT_PATH}_{ITER}_winner.pkl"

    # Save the winner
    with open(winner_path, "wb") as f:
        pickle.dump(winner, f)

    # Save the statistics to a csv file
    with open(STATS_PATH, "a") as f:
        w = csv.writer(f)
        fit_means = stats.get_fitness_mean()
        fit_stdevs = stats.get_fitness_stdev()
        for g in range(len(fit_means)):
            row = (ITER, g, fit_means[g], fit_stdevs[g], best_genomes[g])
            w.writerow(row)

os.environ["SDL_VIDEODRIVER"] = "dummy"

EXPERIMENT_NAME="Custom_generalist"
PLAYER_MODE="ai"
ENEMIES=[1,4,6]
CONTROLLER=generalist()
MULTI_MODE="yes"
SPEED="fastest"
ENEMY_MODE="static"
LEVEL=2
VISUALS=False
GENS=30
ITERATIONS=10
PARALLEL=False
PARALLEL_EVALS=multiprocessing.cpu_count() // 2
NEAT, ESNEAT = False, False

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

best_genomes = []
NEAT, ESNEAT = True, False
ITER = 0

OUT_PATH = f"{EXPERIMENT_NAME}_{''.join(map(str, ENEMIES))}_NEAT"
STATS_PATH = f"stats/{OUT_PATH}_stats.csv"
CONFIG_PATH = 'configs/' + ('neat_generalist.cfg' if NEAT else 'esneat_generalist.cfg' if ESNEAT else 'custom_generalist.cfg')

# Save the statistics to a csv file
with open(STATS_PATH, "w") as f:
    w = csv.writer(f)
    w.writerow(["Iteration", "Generation", "Mean", "Stdev", "Best"])

if __name__ == "__main__":

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'configs/custom_generalist.cfg')

    with ProcessPoolExecutor() as executor:
        executor.map(main, [(config, i) for i in range(ITERATIONS)])