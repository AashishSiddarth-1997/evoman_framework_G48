# statistics.py
# Script to run appropriate statistical tests and make boxplot

# imports 
import csv
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, shapiro

TEST_CSV = "../test.csv"
GEN1 = "7 8"
GEN2 = "2 4"

def read_csv(data_file):
    """
    Returns names of groups (independent variables) and datapoints for each group (dependent variable)
    """
    # initialize all_data and all_groups
    all_data = {}
    # read csv file
    with open(data_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            data_1box = []
            for i in range(3, len(row)-1, 2):
                data_1box.append(float(row[i]))
            all_data[(str(row[0] + row[1]))] = (data_1box)
    return all_data

def boxplot(all_data):
    """
    Makes boxplot and saves it
    """
     # boxplot
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(all_data.values(), patch_artist=True)
    
    # settings
    plt.title("2 EAs, 2 pairs of enemies")
    ax.set_xticklabels(all_data.keys())
    colors = ['gray', 'red', 'green', 'blue', 'cyan']

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    for median in bp['medians']:
        median.set(color ='black',
                    linewidth = 1.5,
                    linestyle ="-")
    # save plot  
    plt.savefig("Boxplot")
    return 

def test(EA1, EA2, all_data):
    """
    Runs a Man Whitney test for EA1 (str label) and EA2 (str label)
    """
    # add shapiro wilks test 
    shapiro1 = shapiro(all_data[EA1])
    shapiro2 = shapiro(all_data[EA2])

    # perform Man Withney
    results = mannwhitneyu(all_data[EA1], all_data[EA2])

    # save results 
    statsfile = open(r"man_whitney_results.csv", "a")
    statsfile.write(EA1 + "," + EA2 + "," + str(results) + str(shapiro1) + str(shapiro2) + '\n')
    return

if __name__ == "__main__":
    # read data 
    data = read_csv(TEST_CSV)

    # make boxplot
    boxplot(data)
    print(data)

    # compare EAs in both generalists
    generalist1 = test(("neat" + GEN1), ("esneat" + GEN1), data)
    generalist2 = test(("neat" + GEN2), ("esneat" + GEN2), data)