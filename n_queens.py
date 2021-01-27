from typing import Dict
from numba import njit
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'binary'


def read_parameters(filename: str) -> Dict[str, float]:
    """Read parameters from a file to a dictionary and return it."""
    parameters = {}
    with open(filename, "r") as file:
        for line in file.readlines():
            if line != '\n':
                line_split = line.split()
                try:
                    parameters[line_split[0]] = int(line_split[2])
                except ValueError:
                    parameters[line_split[0]] = float(line_split[2])
    if len(parameters) != 6:
        raise RuntimeError("Incorrect list of parameters in " + filename)
    return parameters


def random_population(population_size: int, board_size: int) -> np.ndarray:
    """Return a random population of solutions."""
    return np.array([np.random.permutation(board_size)
                     for _ in range(population_size)], dtype=np.int32)


@njit
def fitness(population: np.ndarray) -> np.ndarray:
    """Return an array of fitnesses of a given population"""
    fitness_arr = np.empty(population.shape[0], dtype=np.float32)
    for i, genome in enumerate(population):
        diags_1 = np.array([0 for n in range(2 * genome.size - 1)])
        diags_2 = np.array([0 for n in range(2 * genome.size - 1)])
        for j in range(genome.size):
            diags_1[j - genome[j] + genome.size - 1] += 1
            diags_2[j + genome[j]] += 1
        colls_1 = diags_1 > 1
        colls_2 = diags_2 > 1
        diags_1[colls_1] = diags_1[colls_1] * (diags_1[colls_1] - 1) // 2
        diags_1[~colls_1] = 0
        diags_2[colls_2] = diags_2[colls_2] * (diags_2[colls_2] - 1) // 2
        diags_2[~colls_2] = 0
        fitness_arr[i] = 1 / (1 + np.sum(diags_1) + np.sum(diags_2))
    return fitness_arr


@njit
def selection(population: np.ndarray, n_best: int) -> np.ndarray:
    """Return an array of indices of individuals selected to mate.
    n_best is the number of best individuals who will always be selected.
    """
    fitnesses = fitness(population)
    winners = np.empty((population.shape[0] // 2,), dtype=np.int32)
    winners[0:n_best] = np.argsort(fitnesses)[-n_best:]
    for i in range(n_best, fitnesses.shape[0] // 2):
        pair = np.random.randint(0, fitnesses.shape[0], size=(2,))
        if fitnesses[pair[0]] > fitnesses[pair[1]]:
            winners[i] = pair[0]
        else:
            winners[i] = pair[1]
    return winners


@njit
def crossover(population: np.ndarray, selected: np.ndarray):
    """Return a new population that results from crossover."""
    N = population.shape[1]
    new_population = np.empty_like(population)
    for k in range(0, selected.shape[0]):
        parents_ids = np.random.choice(selected, replace=False, size=2)
        child_1 = np.empty_like(population[parents_ids[0]])
        child_2 = np.empty_like(population[parents_ids[1]])
        points = np.random.randint(0, N + 1, 2)
        if points[0] != points[1]:
            points = (np.min(points), np.max(points))
        else:
            if points[0] == N:
                points = (points[0] - 1, points[0])
            else:
                points = (points[0], points[0] + 1)
        cut_out = population[parents_ids[0]][points[0]:points[1]]
        child_1[points[0]:points[1]] = cut_out
        j = 0
        for i in range(N):
            if j == points[0]:
                j = points[1]
            if not np.any(cut_out == population[parents_ids[1]][i]):
                child_1[j] = population[parents_ids[1]][i]
                j += 1
        cut_out = population[parents_ids[1]][points[0]:points[1]]
        child_2[points[0]:points[1]] = cut_out
        j = 0
        for i in range(N):
            if j == points[0]:
                j = points[1]
            if not np.any(cut_out == population[parents_ids[0]][i]):
                child_2[j] = population[parents_ids[0]][i]
                j += 1
        new_population[2 * k, :] = child_1
        new_population[2 * k + 1, :] = child_2
    return new_population


@njit
def mutation(population: np.ndarray):
    """Perform mutation on a population."""
    for i in range(population.shape[0]):
        if np.random.random() > 0.7:
            for _ in range(3):
                points = np.random.randint(0, population.shape[1], 2)
                tmp = population[i, points[0]]
                population[i, points[0]] = population[i, points[1]]
                population[i, points[1]] = tmp


def plot_genome_expression(genome: np.ndarray) -> None:
    """Plot a solution represented by the given genome."""
    points = np.zeros((genome.shape[0], genome.shape[0]))
    for i, g in enumerate(genome):
        points[i, g] = 1
    _, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(points, cmap='Purples')
    ax.grid(True)
    ax.set_xlim(-0.5, genome.shape[0] - 0.5)
    ax.set_ylim(-0.5, genome.shape[0] - 0.5)
    ax.set_xticks([i + 0.5 for i in range(genome.shape[0])])
    ax.set_yticks([i + 0.5 for i in range(genome.shape[0])])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.tick_params(axis='both', which='both', bottom=False, left=False)
    plt.title("$N = {}$".format(genome.shape[0]), size=15)
    plt.show()


def main() -> None:
    parameters = read_parameters('parameters.txt')
    population = random_population(parameters['pop_size'], parameters['N'])
    generation_data = []
    best_member_id = 0
    winner_gen = parameters['generations']
    for i in range(1, parameters['generations'] + 1):
        selected = selection(population, parameters['n_best'])
        population = crossover(population, selected)
        mutation(population)
        gen_fit = fitness(population)
        best_member_id = np.argmax(gen_fit)
        generation_data.append([i, gen_fit.mean(), gen_fit[best_member_id]])
        if gen_fit[best_member_id] == 1.0:
            print("\nWinner (gen. {}):\n{}".format(
                i, str(population[best_member_id])))
            winner_gen = i
            break
        if i % 50 == 0:
            print("Gen", i)
    if parameters['plot_winner_genome']:
        plot_genome_expression(population[best_member_id])


if __name__ == "__main__":
    main()
