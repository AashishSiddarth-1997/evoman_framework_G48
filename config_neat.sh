[NEAT]
fitness_criterion     = max
fitness_threshold     = 50
pop_size              = 500
reset_on_extinction   = False

[EvomanGenome]
# component type options
component_default      = resistor
component_mutate_rate  = 0.1
component_options      = resistor diode

# component value options
value_init_mean          = 4.5
value_init_stdev         = 0.5
value_max_value          = 6.0
value_min_value          = 3.0
value_mutate_power       = 0.1
value_mutate_rate        = 0.8
value_replace_rate       = 0.1

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 1.0

# connection add/remove rates
conn_add_prob           = 0.2
conn_delete_prob        = 0.2

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.02

# node add/remove rates
node_add_prob           = 0.1
node_delete_prob        = 0.1

# network parameters
num_inputs              = 5
num_outputs             = 1

[DefaultSpeciesSet]
compatibility_threshold = 2.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 15

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2