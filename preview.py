import pickle

NAME = "2358_0"

if __name__ == "__main__":
    with open(f"winners/Custom_generalist_{NAME}_NEAT_winner.pkl", "rb") as f:
        unpickler = pickle.Unpickler(f)
        genome = unpickler.load()
        print(genome)
        print(genome.fitness)

        print(genome.nodes)
        print(len(genome.nodes))

        print(genome.connections)
        print(len(genome.connections))

        with open(f"winners/Custom_generalist_{NAME}_weights.txt", "w") as f:
            for i, weight in enumerate(genome.connections.values()):
                f.write(str(weight.weight))
                if i < len(genome.connections.values()) - 1:
                    f.write('\n')