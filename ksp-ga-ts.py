import numpy as np
import random as rd
from random import randint
import matplotlib.pyplot as plt

dataset_path = "dataset/{}/{}"
item_number = 10
dataset = 15

def read_capacity(path):
    file = open(path, "r")

    content = file.readlines()

    for line in content:
        return int(line)


def read_lines(path):
    file = open(path, "r")

    content = file.readlines()
    lst = []
    for line in content:
        lst.append(int(line))
    return lst


def get_random_dataset():
    nr = np.arange(1, 11)
    weight = np.random.randint(1, 15, size=10)
    value = np.random.randint(10, 750, size=10)
    knapsack_threshold = 35

    return nr, weight, value, knapsack_threshold


def get_dataset(name):
    knapsack_threshold = read_capacity(dataset_path.format(name, "capacity.txt"))
    weight = np.array(read_lines(dataset_path.format(name, "weights.txt")))
    value = np.array(read_lines(dataset_path.format(name, "profits.txt")))

    return weight, value, knapsack_threshold


def print_dataset(weight, value, capacity):
    global item_number
    print("Bag capacity: {}".format(capacity))
    print("The list is as follows:")
    print("Item No.   Weight   Value")
    for i in range(item_number.shape[0]):
        print("{}          {}         {}\n".format(item_number[i], weight[i], value[i]))


def fitness(weight, value, population, threshold):
    fit = np.empty(population.shape[0])
    for i in range(population.shape[0]):
        s1 = np.sum(population[i] * value)
        s2 = np.sum(population[i] * weight)

        if s2 <= threshold:
            fit[i] = s1
        else:
            fit[i] = 0
    return fit.astype(int)


def selection(fit, num_parents, population):
    fit = list(fit)
    parents = np.empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        max_fitness_idx = np.where(fit == np.max(fit))
        parents[i, :] = population[max_fitness_idx[0][0], :]
        fit[max_fitness_idx[0][0]] = -999999
    return parents


def crossover(parents, num_offsprings):
    offsprings = np.empty((num_offsprings, parents.shape[1]))
    crossover_point = int(parents.shape[1] / 2)
    crossover_rate = 0.8
    i = 0
    while parents.shape[0] < num_offsprings:
        x = rd.random()
        if x > crossover_rate:
            continue
        parent1_index = i % parents.shape[0]
        parent2_index = (i + 1) % parents.shape[0]
        offsprings[i, 0:crossover_point] = parents[parent1_index, 0:crossover_point]
        offsprings[i, crossover_point:] = parents[parent2_index, crossover_point:]
        i = +1
    return offsprings


def mutation(offsprings):
    mutants = np.empty(offsprings.shape)
    mutation_rate = 0.4
    for i in range(mutants.shape[0]):
        random_value = rd.random()
        mutants[i, :] = offsprings[i, :]
        if random_value > mutation_rate:
            continue
        int_random_value = randint(0, offsprings.shape[1] - 1)
        if mutants[i, int_random_value] == 0:
            mutants[i, int_random_value] = 1
        else:
            mutants[i, int_random_value] = 0
    return mutants


def optimize(weight, value, population, pop_size, num_generations, threshold):
    parameters, fitness_history = [], []
    num_parents = int(pop_size[0] / 2)
    num_offsprings = pop_size[0] - num_parents

    for i in range(num_generations):
        fit = fitness(weight, value, population, threshold)
        fitness_history.append(fit)
        parents = selection(fit, num_parents, population)
        offsprings = crossover(parents, num_offsprings)
        mutants = mutation(offsprings)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants

    print('Last generation: \n{}\n'.format(population))

    fitness_last_gen = fitness(weight, value, population, threshold)
    print('Fitness of the last generation: \n{}\n'.format(fitness_last_gen))

    max_fitness = np.where(fitness_last_gen == np.max(fitness_last_gen))
    parameters.append(population[max_fitness[0][0], :])

    return parameters, fitness_history


def visualize(num_generations, fitness_history):
    fitness_history_mean = [np.mean(fit) for fit in fitness_history]
    fitness_history_max = [np.max(fit) for fit in fitness_history]

    plt.plot(list(range(num_generations)), fitness_history_mean, label='Mean Fitness')
    plt.plot(list(range(num_generations)), fitness_history_max, label='Max Fitness')

    plt.legend()

    plt.title('Fitness through the generations')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')

    plt.show()

    print(np.asarray(fitness_history).shape)


def main():
    global item_number

    weight, value, knapsack_threshold = get_dataset(dataset)
    item_number = np.arange(1, int(str(dataset).split("_")[0]) + 1)

    # item_number, weight, value, knapsack_threshold = get_random_dataset()

    print_dataset(weight, value, knapsack_threshold)

    solutions_per_pop = 1000
    pop_size = (solutions_per_pop, item_number.shape[0])
    print('Population size = {}'.format(pop_size))
    initial_population = np.random.randint(2, size=pop_size)
    initial_population = initial_population.astype(int)
    num_generations = 50
    print('Initial population: \n{}'.format(initial_population))


    parameters, fitness_history = optimize(weight, value, initial_population, pop_size, num_generations,
                                           knapsack_threshold)

    print('The optimized parameters for the given inputs are: \n{}'.format(parameters))
    selected_items = item_number * parameters
    print('\nSelected items that will maximize the knapsack without breaking it:')
    for i in range(selected_items.shape[1]):
        if selected_items[0][i] != 0:
            print('{}\n'.format(selected_items[0][i]))

    # visualize(num_generations, fitness_history)


if __name__ == "__main__":
    main()
