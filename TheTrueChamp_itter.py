import sys
from evoman.environment import Environment
from demo_controller import player_controller
import math
import time
import numpy as np
import glob, os
import matplotlib.pyplot as plt
from scipy.stats import f_oneway


#for k in range(1,9):
def run_ea_once(p):
    
    experiment_name = "Enemy_" + str(1)
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10
    ini = time.time()
    population_number = 100
    generations = 30
    last_best =0
    run_mode ='train'

    def fitness(env,x):
        fit,p_life,enemy_life,times = env.play(pcont=x)
        f = 0.9*(100 - enemy_life) + 0.1*p_life - np.log(times)
        if p_life < 0.2 :
            f = f - 10
        if enemy_life > 0.8:
            f = f - 10
        if  times > 10:
            f = f - 1
        return max(f,0)

    def evaluate(x):
        return np.array(list(map(lambda y:fitness(env,y),x)))

    #Tournoua lol  1 vs 1 zed mid 
    def Arena(popsize,fitness_score,g):
        
            Arena_bracket = np.random.choice(len(popsize), size=2)
            P1,P2 = popsize[Arena_bracket]
            if g <= 9 :
                f1_score,f2_score = fitness_score[Arena_bracket]
                winner = Arena_bracket[np.argmax([f1_score,f2_score])]
                the_chosen_one = popsize[winner]
            else:
                if np.random.rand() < 0.5:
                    the_chosen_one = P1
                else:
                    the_chosen_one = P2
            
            
            the_chosen_one = np.asarray(the_chosen_one)
            return the_chosen_one

    #Permutation Represendation Order 1 Knights of Gondor Crossover  + self adapt mutation
    def crossover(popsize,fitness_score,mutation_rate,g):
        
        parents_num = len(popsize)//2
        genes_num = len(popsize[0])
        children = []
        best_fitness = max(fitness_score)
        worst_fitness = min(fitness_score)
        mutation_strength = 0.2 * (best_fitness - worst_fitness)

        for i in range(parents_num):
            p1 = Arena(popsize,fitness_score,g)
            p2 = Arena(popsize,fitness_score,g)
            crossover_p1, crossover_p2 = np.random.choice(range(genes_num), size=2 ,replace=False)
            if crossover_p1 > crossover_p2:
                maxp = crossover_p1
                minp = crossover_p2
            else:
                maxp = crossover_p2
                minp = crossover_p1

            child = []

            for i in range(genes_num):
                if  minp>= i and  maxp<=i :
                    child.append(p1[i])
                else:
                    child.append(p2[i])

            children.append(child)

        children_arr = np.array(children)
        #mutans
        for i in range(parents_num):
            for j in range(n_vars):
                    if np.random.uniform(0, 1) < mutation_rate:
                    # Mutate the gene by adding a small random value
                        mutation_value = np.random.normal(0, mutation_strength)
                        children_arr[i][j] += mutation_value

        return children_arr

    def destruction(pop, fit_pop, cull_perc,g):
        num_of_cull = int(len(pop) * cull_perc)
        surv_number = len(pop) - num_of_cull
        survivors = []

        for i in range(surv_number):
            points_of_indices = np.random.choice(len(pop), size=2, replace=False)
            idx1, idx2 = points_of_indices
            if fit_pop[idx1] < fit_pop[idx2]:
                survivors.append(pop[idx2])
            else:
                survivors.append(pop[idx1])

        best = np.argmax(fit_pop)
        surv_fit =evaluate(survivors)
        #print(fitness(env,pop[best]))
        survivors = np.array(survivors)

        old_renew =[]
        for i in range(num_of_cull):
            neww=Arena(survivors,surv_fit,g)
            old_renew.append(neww)
        old_renew = np.array(old_renew)

        # Iterate through each individual and each gene to apply mutation
        for i in range(len(old_renew)):
            for j in range(n_vars):
                    # Mutate the gene by replacing it with a small random value
                    mutation_value = np.random.normal(0, 0.2)
                    old_renew[i][j] += mutation_value

        survivors=np.concatenate((survivors,old_renew),axis=0)
        New_fit_pop = evaluate(survivors)

        return survivors, New_fit_pop

    def individual_gain(env,x):
        L,P_life,Enemy_life,t = env.play(x)
        indi_gain = P_life - Enemy_life
        return indi_gain

    env=Environment(experiment_name=experiment_name,
            enemies=[1],
            playermode='ai',
            player_controller=player_controller(n_hidden_neurons),
            enemymode="static",
            level=2,
            speed='fastest',
            visuals=True)

    if run_mode=='test':
        best_sol = np.loadtxt(experiment_name+'/best.txt')
        print('\n Running save with best solution\n')
        env.update_parameter('speed','normal')
        env.update_parameter('visuals',str(True))
        Feed_back =individual_gain(env,best_sol)
        print('This is the individual gain -> ',Feed_back)
        sys.exit(0)
    else:
        headless = True
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

    n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

    #if not os.path.exists(experiment_name+'/evoman_solstate'):

    print( '\nNEW EVOLUTION\n')
    population = np.random.uniform(-1,1,(population_number,n_vars))
    fit_population = evaluate(population)
    best = np.argmax(fit_population)
    mean = np.mean(fit_population)
    std =np.std(fit_population)
    ini_g = 0
    solutions = [population, fit_population]
    env.update_solutions(solutions)


    best_fit_all = fit_population
    last_solution = fit_population[best]
    stacking = 0
    best_of_all = best

    for i in range(generations):
        #create new kiddos
        offsping = crossover(population,fit_population,0.2,stacking)
        fit_offsping = evaluate(offsping)
        worst_population = np.argsort(fit_population)[:len(offsping)]
        population[worst_population] = offsping
        fit_population[worst_population] = fit_offsping

        best = np.argmax(fit_population)
        mean = np.mean(fit_population)
        std =np.std(fit_population)
    
        #save the  BD King
        best_sol = fit_population[best]
        if best_fit_all[best_of_all] <= best_sol :
            stacking = 0
            best_of_all = best
            g = i
            best_p = population
            best_fit_all = fit_population
            solutions=[best_p,best_fit_all]
            env.update_solutions(solutions)
            env.save_state()
            np.savetxt(experiment_name+'/best' + str(p) + '.txt',population[best])
        else:
            stacking+=1

        if stacking >= 10:
            stacking = 0
            file_aux  = open(experiment_name+'/results.txt','a')
            file_aux.write('\ndestruction')
            file_aux.close()
            population,fit_population = destruction(population,fit_population,0.5,stacking)
    # saves results
        file_aux  = open(experiment_name+'/results.txt','a')
        print( '\n GENERATION '+str(i)+' '+str(round(fit_population[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
        file_aux.write('\n'+str(i)+' '+str(round(fit_population[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
        file_aux.close()
    
    fim = time.time() # prints total execution time for experiment
    print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
    print( '\nExecution time: '+str(round((fim-ini)))+' seconds \n')
    print('\nBest solution was on '+str(g)+' Generation','','and its :',str(round(best_fit_all[best_of_all],6)),'\n')

#env.state_to_log()
# Call the function to run your EA
#average_fitness = run_ea_once()



# Initialize arrays to store average fitness values
avg_fitness_values = []

avg_fitness_per_run = []
for p in range(1,11):
        # Run your EA code here and obtain the average fitness for this run
        average_fitness = run_ea_once(p)
        avg_fitness_per_run.append(average_fitness)
    
    # Store the average fitness values for this repetition
avg_fitness_values.append(avg_fitness_per_run)

# Convert to a NumPy array for easier manipulation
avg_fitness_values = np.array(avg_fitness_values)

    # Perform ANOVA to test for significant differences
    # You'll need to adapt this to your specific data structure
    # Each column represents a combination of task, algorithm, and enemy/group
    # Make sure to reshape your data accordingly
    # f_statistic, p_value = f_oneway(avg_fitness_values[:, 0], avg_fitness_values[:, 1], ...)
    # f_statistic, p_value = f_oneway(avg_fitness_values[:, 0], avg_fitness_values[:, 1], avg_fitness_values[:, 2], ...)






    # Assuming avg_fitness_values is a list of NumPy arrays containing your data
    # Replace '...' with the actual arrays you want to compare
    #f_statistic, p_value = f_oneway(avg_fitness_values[0], avg_fitness_values[1], avg_fitness_values[2], ...)



    #f_statistic, p_value = f_oneway(*avg_fitness_values)





# Plot boxplots for each combination
# plt.boxplot(avg_fitness_values)
# plt.xlabel('Combinations')
# plt.ylabel('Average Fitness')
# plt.title('Boxplot of Average Fitness Values')
# plt.show()

# Print the ANOVA test result
# if p_value < 0.05:  # Adjust the significance level as needed
#     print("Differences in average fitness values are statistically significant.")
# else:
#     print("No statistically significant differences observed.")


