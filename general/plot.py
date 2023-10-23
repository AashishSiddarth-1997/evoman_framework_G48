import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import numpy as np
import sys
import csv

def line_plot(all_max_gens, all_mean_gens, max_std_lower, mean_std_lower, max_std_upper, mean_std_upper, names, plot_name):
    # print("Plotting...")
    # print("all_max_gens: ", all_max_gens)
    # print("all_mean_gens: ", all_mean_gens)
    # print("max_std_lower: ", max_std_lower)
    # print("mean_std_lower: ", mean_std_lower)
    # print("max_std_upper: ", max_std_upper)
    # print("mean_std_upper: ", mean_std_upper)
    # print("names: ", names)
    n = 0
    # for n in range(1):
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title(f"Group {plot_name} line plot of best and mean performance",size=10)
    plt.xlim(0, 24)
    plt.ylim(-10, 100)
    # print(all_max_gens)
    # print(all_mean_gens)
    for i in range(2):
        plt.plot(all_max_gens[i+n], color=None, alpha=1.0,label="Best " + names[i+n])
        plt.plot(all_mean_gens[i+n], color=None, alpha=1.0,label="Mean " + names[i+n])
        plt.fill_between(x=range(25), y1=max_std_lower[i+n],y2=max_std_upper[i+n], alpha=.2)
        plt.fill_between(x=range(25), y1=mean_std_lower[i+n],y2=mean_std_upper[i+n], alpha=.2)
    plt.legend(loc="best",prop={'size': 5})
    plt.savefig(plot_name)
    # plt.show()
    plt.clf()

ITERS = [0, 1, 2, 3, 4, 6, 7, 8]
GENS = 25
  
if __name__ == "__main__":
    for g in ['146', '2358']:
        # Gather paths
        paths = glob(r"stats/*{}*".format(g))
        # Initialize std lists
        std_mean_gens1=[]
        std_max_gens1=[]
        # Initialize mean/max lists
        all_mean_gens1=[]
        all_max_gens1=[]
        # Initialize upper/lower std lists
        mean_std_upper1 = []
        mean_std_lower1 = []
        max_std_upper1 = []
        max_std_lower1 = []
        # Initialize a list of names
        names=[]

        for i in paths:
            name = (i.split("/"))[1]

            if "ESNEAT" in name:
                NEAT, ESNEAT = False, True
            else:
                NEAT, ESNEAT = True, False
            
            group = "".join([c for c in name if c.isdigit()])

            names.append(f"{'NEAT' if NEAT else 'HYPERNEAT'} {group}")

            df = pd.read_csv(f"/Users/duculet/evoman_framework_G48/stats/{name}")
            
            mean_gens=[]
            max_gens=[]

            print("*" * 50)
            print(name)
            print(df["Iteration"].unique())
            print("*" * 50)

            for i in ITERS:
                temp=[]
                temp_1=[]

                # mean = df[df['Iteration'] == i]

                # print(mean)


                for g in range(GENS):
                    # only keep the value in row with Iteration = i and Generation = g
                    mean = df[(df['Iteration'] == i) & (df['Generation'] == g)]['Mean']
                    if not mean.any():
                        print("No mean value found")
                        print("Iteration: ", i)
                        print("Generation: ", g)
                    mean = mean.values[0]
                    # print("_" * 50)
                    # print(mean)
                    # print("_" * 50)
                    temp.append(mean)
                    maxg = df[(df['Iteration'] == i) & (df['Generation'] == g)]['Best'].values[0]
                    # print("_" * 50)
                    # print(maxg)
                    # print("_" * 50)
                    temp_1.append(maxg)
                    # break
                mean_gens.append(temp)
                max_gens.append(temp_1)
                # break
            
            # break

            all_mean_gens = np.array(mean_gens)
            all_max_gens = np.array(max_gens)

            std_mean_gens1.append(np.std(all_mean_gens, axis=0))
            std_max_gens1.append(np.std(all_max_gens, axis=0))
            print("-" * 50)
            print("ADDING TO LIST", i)
            all_mean_gens1.append(np.mean(all_mean_gens, axis=0))
            all_max_gens1.append(np.mean(all_max_gens, axis=0))
            print(all_mean_gens1)
            print("-" * 50)

            mean_std_upper1.append(np.mean(all_mean_gens, axis=0) + np.std(all_mean_gens, axis=0))
            mean_std_lower1.append(np.mean(all_mean_gens, axis=0) - np.std(all_mean_gens, axis=0))

            max_std_upper1.append(np.mean(all_max_gens, axis=0) + np.std(all_max_gens, axis=0))
            max_std_lower1.append(np.mean(all_max_gens, axis=0) - np.std(all_max_gens, axis=0))

            print("Plotting...")
            # print("all_max_gens: ", all_max_gens)
            # print("all_mean_gens: ", all_mean_gens)
            # print("max_std_lower: ", max_std_lower1)
            # print("mean_std_lower: ", mean_std_lower1)
            # print("max_std_upper: ", max_std_upper1)
            # print("mean_std_upper: ", mean_std_upper1)
            print("names: ", names)
        line_plot(all_max_gens1, all_mean_gens1, max_std_lower1, mean_std_lower1, max_std_upper1, mean_std_upper1, names, group)
        