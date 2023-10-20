import pickle

NAME = "2358"
ITERATIONS = 10

if __name__ == "__main__":
    for i in range(ITERATIONS):
        with open(f"winners/Custom_generalist_{NAME}_ESNEAT_{i}_winner.pkl", "rb") as f:
            unpickler = pickle.Unpickler(f)
            genome = unpickler.load()
            print("Iteration", i)
            print("Fitness", genome.fitness)

            # print(genome.nodes)
            # print(len(genome.nodes))

            # print(genome.connections)
            # print(len(genome.connections))

            # with open(f"winners/Custom_generalist_{NAME}_weights.txt", "w") as f:
            #     for i, weight in enumerate(genome.connections.values()):
            #         f.write(str(weight.weight))
            #         if i < len(genome.connections.values()) - 1:
            #             f.write('\n')