import os
import time
from random import choice, random

import numpy as np
import matplotlib.pyplot as plt

from evoman.environment import Environment
from evoman.controller import Controller
from demo_controller import player_controller 

from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit

import neat
from neat.activations import ActivationFunctionSet
from neat.attributes import FloatAttribute, BoolAttribute, StringAttribute
from neat.config import ConfigParameter, write_pretty_params
from neat.genes import BaseGene
from neat.six_util import iteritems, iterkeys

class EvomanNodeGene(BaseGene):
    __gene_attributes__ = []

    def distance(self, other, config):
        return 0.0


class EvomanConnectionGene(BaseGene):
    __gene_attributes__ = [StringAttribute('component'),
                           FloatAttribute('value'),
                           BoolAttribute('enabled')]

    def distance(self, other, config):
        d = abs(self.value - other.value)
        if self.component != other.component:
            d += 1.0
        if self.enabled != other.enabled:
            d += 1.0
        return d * config.compatibility_weight_coefficient


class EvomanGenomeConfig(object):
    __params = [ConfigParameter('num_inputs', int),
                ConfigParameter('num_outputs', int),
                ConfigParameter('compatibility_disjoint_coefficient', float),
                ConfigParameter('compatibility_weight_coefficient', float),
                ConfigParameter('conn_add_prob', float),
                ConfigParameter('conn_delete_prob', float),
                ConfigParameter('node_add_prob', float),
                ConfigParameter('node_delete_prob', float)]

    def __init__(self, params):
        # Create full set of available activation functions.
        self.activation_defs = ActivationFunctionSet()
        self.activation_options = params.get('activation_options', 'sigmoid').strip().split()
        self.aggregation_options = params.get('aggregation_options', 'sum').strip().split()

        # Gather configuration data from the gene classes.
        self.__params += EvomanNodeGene.get_config_params()
        self.__params += EvomanConnectionGene.get_config_params()

        # Use the configuration data to interpret the supplied parameters.
        for p in self.__params:
            setattr(self, p.name, p.interpret(params))

        # By convention, input pins have negative keys, and the output
        # pins have keys 0,1,...
        self.input_keys = [-i - 1 for i in range(self.num_inputs)]
        self.output_keys = [i for i in range(self.num_outputs)]

    def save(self, f):
        write_pretty_params(f, self, self.__params)


class EvomanGenome(object):
    @classmethod
    def parse_config(cls, param_dict):
        return EvomanGenomeConfig(param_dict)

    @classmethod
    def write_config(cls, f, config):
        config.save(f)

    def __init__(self, key):
        self.key = key

        # (gene_key, gene) pairs for gene sets.
        self.connections = {}
        self.nodes = {}

        # Fitness results.
        self.fitness = None

    def mutate(self, config):
        """ Mutates this genome. """

        # TODO: Make a configuration item to choose whether or not multiple
        # mutations can happen simultaneously.
        if random() < config.node_add_prob:
            self.mutate_add_node(config)

        if random() < config.node_delete_prob:
            self.mutate_delete_node(config)

        if random() < config.conn_add_prob:
            self.mutate_add_connection(config)

        if random() < config.conn_delete_prob:
            self.mutate_delete_connection()

        # Mutate connection genes.
        for cg in self.connections.values():
            cg.mutate(config)

        # Mutate node genes (bias, response, etc.).
        for ng in self.nodes.values():
            ng.mutate(config)

    def configure_crossover(self, genome1, genome2, config):
        """ Configure a new genome by crossover from two parent genomes. """
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1

        # Inherit connection genes
        for key, cg1 in iteritems(parent1.connections):
            cg2 = parent2.connections.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.connections[key] = cg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.connections[key] = cg1.crossover(cg2)

        # Inherit node genes
        parent1_set = parent1.nodes
        parent2_set = parent2.nodes

        for key, ng1 in iteritems(parent1_set):
            ng2 = parent2_set.get(key)
            assert key not in self.nodes
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                self.nodes[key] = ng1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.nodes[key] = ng1.crossover(ng2)

    def get_new_node_key(self):
        new_id = 0
        while new_id in self.nodes:
            new_id += 1
        return new_id

    def mutate_add_node(self, config):
        if not self.connections:
            return None, None

        # Choose a random connection to split
        conn_to_split = choice(list(self.connections.values()))
        new_node_id = self.get_new_node_key()
        ng = self.create_node(config, new_node_id)
        self.nodes[new_node_id] = ng

        # Disable this connection and create two new connections joining its nodes via
        # the given node.  The new node+connections have roughly the same behavior as
        # the original connection (depending on the activation function of the new node).
        conn_to_split.enabled = False

        i, o = conn_to_split.key
        self.add_connection(config, i, new_node_id)
        self.add_connection(config, new_node_id, o)

    def add_connection(self, config, input_key, output_key):
        # TODO: Add validation of this connection addition.
        key = (input_key, output_key)
        connection = EvomanConnectionGene(key)
        connection.init_attributes(config)
        self.connections[key] = connection

    def mutate_add_connection(self, config):
        """
        Attempt to add a new connection, the only restriction being that the output
        node cannot be one of the network input pins.
        """
        possible_outputs = list(iterkeys(self.nodes))
        out_node = choice(possible_outputs)

        possible_inputs = possible_outputs + config.input_keys
        in_node = choice(possible_inputs)

        if in_node == out_node:
            return

        # # Don't duplicate connections.
        # key = (in_node, out_node)
        # if key in self.connections:
        #     return

        cg = self.create_connection(config, in_node, out_node)
        self.connections[cg.key] = cg

    def mutate_delete_node(self, config):
        # Do nothing if there are no non-output nodes.
        available_nodes = [(k, v) for k, v in iteritems(self.nodes) if k not in config.output_keys]
        if not available_nodes:
            return -1

        del_key, del_node = choice(available_nodes)

        connections_to_delete = set()
        for k, v in iteritems(self.connections):
            if del_key in v.key:
                connections_to_delete.add(v.key)

        for key in connections_to_delete:
            del self.connections[key]

        del self.nodes[del_key]

        return del_key

    def mutate_delete_connection(self):
        if self.connections:
            key = choice(list(self.connections.keys()))
            del self.connections[key]

    def distance(self, other, config):
        """
        Returns the genetic distance between this genome and the other. This distance value
        is used to compute genome compatibility for speciation.
        """

        # Compute node gene distance component.
        node_distance = 0.0
        if self.nodes or other.nodes:
            disjoint_nodes = 0
            for k2 in iterkeys(other.nodes):
                if k2 not in self.nodes:
                    disjoint_nodes += 1

            for k1, n1 in iteritems(self.nodes):
                n2 = other.nodes.get(k1)
                if n2 is None:
                    disjoint_nodes += 1
                else:
                    # Homologous genes compute their own distance value.
                    node_distance += n1.distance(n2, config)

            max_nodes = max(len(self.nodes), len(other.nodes))
            node_distance = (node_distance + config.compatibility_disjoint_coefficient * disjoint_nodes) / max_nodes

        # Compute connection gene differences.
        connection_distance = 0.0
        if self.connections or other.connections:
            disjoint_connections = 0
            for k2 in iterkeys(other.connections):
                if k2 not in self.connections:
                    disjoint_connections += 1

            for k1, c1 in iteritems(self.connections):
                c2 = other.connections.get(k1)
                if c2 is None:
                    disjoint_connections += 1
                else:
                    # Homologous genes compute their own distance value.
                    connection_distance += c1.distance(c2, config)

            max_conn = max(len(self.connections), len(other.connections))
            connection_distance = (connection_distance + config.compatibility_disjoint_coefficient * disjoint_connections) / max_conn

        distance = node_distance + connection_distance

        return distance

    def size(self):
        """Returns genome 'complexity', taken to be (number of nodes, number of enabled connections)"""
        num_enabled_connections = sum([1 for cg in self.connections.values() if cg.enabled is True])
        return len(self.nodes), num_enabled_connections

    def __str__(self):
        s = "Nodes:"
        for k, ng in iteritems(self.nodes):
            s += "\n\t{0} {1!s}".format(k, ng)
        s += "\nConnections:"
        connections = list(self.connections.values())
        connections.sort()
        for c in connections:
            s += "\n\t" + str(c)
        return s

    def add_hidden_nodes(self, config):
        for i in range(config.num_hidden):
            node_key = self.get_new_node_key()
            assert node_key not in self.nodes
            node = self.__class__.create_node(config, node_key)
            self.nodes[node_key] = node

    def configure_new(self, config):
        # Create node genes for the output pins.
        for node_key in config.output_keys:
            self.nodes[node_key] = self.create_node(config, node_key)

        for input_id in config.input_keys:
            for node_id in iterkeys(self.nodes):
                connection = self.create_connection(config, input_id, node_id)
                self.connections[connection.key] = connection

    @staticmethod
    def create_node(config, node_id):
        node = EvomanNodeGene(node_id)
        node.init_attributes(config)
        return node

    @staticmethod
    def create_connection(config, input_id, output_id):
        connection = EvomanConnectionGene((input_id, output_id))
        connection.init_attributes(config)
        return connection


def get_pins(key):
    pins = []
    for k in key:
        if k < 0:
            pins.append('input{0}'.format(-k))
        else:
            pins.append('node{0}'.format(k))

    return pins


def create_evoman(genome, config):
    env = Environment(experiment_name=E_NAME,
                        enemymode='static',
                        speed="fastest", # or normal
                        sound="off",
                        fullscreen=False,
                        use_joystick=True,
                        playermode='ai',
                        visuals=False)
    env.update_parameter('enemies', [EN])
    env.play()


    libraries_path = '/home/alan/ngspice/libraries'  # os.path.join(os.path.dirname(os.path.dirname(__file__)), 'libraries')
    spice_library = SpiceLibrary(libraries_path)

    circuit = Circuit('NEAT')
    circuit.include(spice_library['1N4148'])

    Vbase = circuit.V('base', 'input1', circuit.gnd, 2)
    Vcc = circuit.V('cc', 'input2', circuit.gnd, 5)
    Vgnd = circuit.V('gnd', 'input3', circuit.gnd, 0)
    #circuit.R('test1', 'node0', circuit.gnd, 1e6)
    #circuit.R('test2', 'node0', 'input1', 1e6)
    ridx = 1
    xidx = 1
    for key, c in iteritems(genome.connections):
        if c.component == 'left':
            c.value = 0
        elif c.component == 'diode':
            pin0, pin1 = get_pins(key)
            circuit.X(xidx, '1N4148', pin1, pin0)
            xidx += 1

    return circuit

# implements controller structure for player
class player_controller(Controller):
	def __init__(self, _n_hidden):
		self.n_hidden = [_n_hidden]

	def set(self, controller, n_inputs):
		# Number of hidden neurons

		if self.n_hidden[0] > 0:
			# Preparing the weights and biases from the controller of layer 1

			# Biases for the n hidden neurons
			self.bias1 = controller[:self.n_hidden[0]].reshape(1, self.n_hidden[0])
			# Weights for the connections from the inputs to the hidden nodes
			weights1_slice = n_inputs * self.n_hidden[0] + self.n_hidden[0]
			self.weights1 = controller[self.n_hidden[0]:weights1_slice].reshape((n_inputs, self.n_hidden[0]))

			# Outputs activation first layer.


			# Preparing the weights and biases from the controller of layer 2
			self.bias2 = controller[weights1_slice:weights1_slice + 5].reshape(1, 5)
			self.weights2 = controller[weights1_slice + 5:].reshape((self.n_hidden[0], 5))

	def control(self, inputs, controller):
		# Normalises the input using min-max scaling
		inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))

		if self.n_hidden[0]>0:
			# Preparing the weights and biases from the controller of layer 1

			# Outputs activation first layer.
			output1 = sigmoid_activation(inputs.dot(self.weights1) + self.bias1)

			# Outputting activated second layer. Each entry in the output is an action
			output = sigmoid_activation(output1.dot(self.weights2)+ self.bias2)[0]
		else:
			bias = controller[:5].reshape(1, 5)
			weights = controller[5:].reshape((len(inputs), 5))

			output = sigmoid_activation(inputs.dot(weights) + bias)[0]

		# takes decisions about sprite actions
		if output[0] > 0.5:
			left = 1
		else:
			left = 0

		if output[1] > 0.5:
			right = 1
		else:
			right = 0

		if output[2] > 0.5:
			jump = 1
		else:
			jump = 0

		if output[3] > 0.5:
			shoot = 1
		else:
			shoot = 0

		if output[4] > 0.5:
			release = 1
		else:
			release = 0

		return [left, right, jump, shoot, release]


class NeatController(Controller):
    def __init__(self) -> None:
        super().__init__()

    def set(self, genome, n_inputs):
        super().set(genome, n_inputs)
    
    def control(self, inputs, controller):
        # control the values of the inputs
        super().control(inputs, controller)

def simulate(genome, config):
    try:
        # for key, c in iteritems(genome.connections):
             
        #     if c.component == 'left':
        #          return [1, 0, 0, 0, 0]
        #     elif c.component == 'right':
        #         return [0, 1, 0, 0, 0]
        #     elif c.component == 'jump':
        #         return [0, 0, 1, 0, 0]
        #     elif c.component == 'shoot':
        #         return [0, 0, 0, 1, 0]
        #     elif c.component == 'release':
        #         return [0, 0, 0, 0, 1]
        #     else:
        #         raise RuntimeError("Unknown component: " + c.component)
            
        # global EN
        # neat_controller = NeatController()
        # neat_controller.set(genome, config.genome_config.num_inputs)

        env = Environment(experiment_name=E_NAME,
                        enemymode='static',
                        speed="fastest", # or normal
                        sound="off",
                        fullscreen=False,
                        use_joystick=True,
                        playermode='ai',
                        player_controller=player_controller(N_HIDDEN),
                        visuals=False)
        
        env.update_parameter('enemies', [EN])
        env.play()
        
        fitness = round(env.fitness_single(), 2)

        return fitness
    except Exception as e:
        print(e)
        return -10.0


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = simulate(genome, config)

# def save():
#     file_save_res = f"testLogs/fitnesses_{time.strftime('%H:%M:%S')}.csv"
#     with open(file_save_res, 'w') as f:
#         csv_writer = csv.writer(f, dialect='excel')
#         header = ['Enemy', 'Iteration', 'Generation', 'Fitness', 'Mean', 'Best']
#         csv_writer.writerow(header)
#         for i in range(ITERATIONS):
#             for en in ENEMIES:
#                 for gen in range(GENERATIONS):

#                     val_fit = fitness_over_iterations[i][en]["Generations"][gen]
#                     val_best = fitness_over_iterations[i][en]["Best"]
#                     val_mean = fitness_over_iterations[i][en]["Mean"]

#                     row = [en, i + 1, gen + 1, val_fit, val_mean, val_best]
#                     csv_writer.writerow(row)

def create_evoman(genome, config):
    # Load configuration.
    config = neat.Config(genome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config)
    #config.save('test_save_config.txt')

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    neat_controller = NeatController()

    env = Environment(experiment_name=E_NAME,
                        level=2,
                        enemymode='static',
                        speed="fastest", # or normal
                        sound="off",
                        fullscreen=False,
                        use_joystick=True,
                        playermode='ai',
                        player_controller=neat_controller,
                        visuals=False)
    

    return env


def run(config_file):
    # Load configuration.
    config = neat.Config(EvomanGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    #config.save('test_save_config.txt')

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 30 generations.
    pe = neat.ParallelEvaluator(4, simulate)
    p.run(pe.evaluate, GENERATIONS)

    # Write run statistics to file.
    stats.save()

    # Display the winning genome.
    winner = stats.best_genome()
    print('\nBest genome:\nfitness {!s}\n{!s}'.format(winner.fitness, winner))

    # One final RUN with winner
    winner_evoman = create_evoman(winner, config)

    winner_evoman.update_parameter('enemies', [en])
    winner_evoman.play()

    print(winner_evoman)

    fitness = winner_evoman.fitness_single()

    print("Final BOSS fitness", fitness)

    # plt.plot(inputs, outputs, 'r-', label='output')
    # plt.plot(inputs, expected, 'g-.', label='target')
    # plt.grid()
    # plt.legend(loc='best')
    # plt.gca().set_aspect(1)
    # plt.show()

    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=False)

E_NAME = 'evoman'
CFG = 'config_neat.sh'
N_HIDDEN = 10
ITERATIONS = 10
ITER = 0
GENERATIONS = 30
GEN = 1
ENEMIES = [1, 2, 4]
EN = 1

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    print(local_dir, os.getcwd())

    config_path = os.path.join(local_dir, CFG)
    run(config_path)
    # for en in ENEMIES:
    #     EN = en
    #     for i in range(ITERATIONS):
    #         ITER = i
    #         print("Iteration:", i)
    #         run(config_path)