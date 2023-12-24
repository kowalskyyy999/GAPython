from typing import Any
from src.gen import generate_gen
from src.calc_fitness import calculation, calculation_mutation

import control
from control import TransferFunction
from tqdm import tqdm
from random import random
import numpy as np 
import copy
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    def __init__(self, num, den, n_var, n_bit, ra, rb, jumlah_populasi, minimum_target = 75) -> None:
        self.num = np.array(num) 
        self.den = np.array(den)
        self.n_var = n_var
        self.n_bit = n_bit
        self.ra = ra 
        self.rb = rb
        self.jumlah_populasi = jumlah_populasi
        self.target = minimum_target

    def create_populasi(self):
        print("Create the population")
        population = []
        for _ in tqdm(range(self.jumlah_populasi)):
            gen, kromosom = generate_gen(self.n_var, self.n_bit, self.ra, self.rb)
            fitness = calculation(self.num, self.den, gen)
            population.append(
                {
                    "gen": gen,
                    "fitness": fitness,
                    "kromosom": kromosom
                }
            )
        return population
    
    def selection(self, population):
        population_cp = copy.deepcopy(population)
        fitness = []
        for pop in population_cp:
            fitness.append(pop["fitness"])
        index = np.argmax(fitness)
        parent1 = population_cp[index]

        population_cp[index] = {}
        fitness[index] = 0

        index = np.argmax(fitness)
        parent2 = population_cp[index]

        return parent1, parent2

    def crossover(self, parent1, parent2):
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        slice_index = len(parent1["kromosom"]) // 2

        child1["kromosom"][:slice_index] = parent2["kromosom"][:slice_index]
        child2["kromosom"][:slice_index] = parent1["kromosom"][:slice_index]

        return child1, child2
    
    def mutation(self, child, mutation_rate):
        mutant = child
        mutator = {
            0: 1, 
            1: 0
        }
        kromosom = child["kromosom"]
        for i in range(len(kromosom)):
            if random() < mutation_rate:
                kromosom[i] = mutator[kromosom[i]]

        mutant["kromosom"] = kromosom

        return mutant
    
    def regeneration(self, children, population):
        fitness = []
        for pop in population:
            fitness.append(pop["fitness"])

        for i in range(len(children)):
            index = np.argmin(fitness)
            population[index] = children[i]
            fitness[index] = np.inf

        return population
    
    def termination(self, population):
        best, _ = self.selection(population)

        if best["fitness"] > self.target:
            loop = False
        else: 
            loop = True

        return best, loop
    
    def display_out(self, pop, generation):
        print("##### Optimizing PID using Genetic Algorithm #####")
        print(f"* Generation: {generation}")
        print(f"* KP        : {pop['gen'][0]}")
        print(f"* KI        : {pop['gen'][1]}")
        print(f"* KD        : {pop['gen'][2]}")
        print(f"* Fitness   : {pop['fitness']}")

    def get_PID(self, num, den, pop):
        tf_sys = TransferFunction(num, den)
        best_gen_pid = pop["gen"]
        kp, ki, kd = best_gen_pid[0], best_gen_pid[1], best_gen_pid[2]

        tf_pid = TransferFunction(np.array([kp, ki, kd]), [1, 0])

        tf_mul = tf_sys * tf_pid
        tf_sys_pid = tf_mul.feedback()

        response = control.step_response(tf_sys_pid)
        result = control.step_info(tf_sys_pid)
        
        print("### Best of PID Parameters ###")
        print(f"---> KP: {kp} - KI: {ki} - KD: {kd} <---")
        print("@@ Result:")
        print(f"* Rise Time         : {result['RiseTime']}")
        print(f"* Overshoot         : {result['Overshoot']}")
        print(f"* SettlingTime      : {result['SettlingTime']}")
        print(f"* SteadyState       : {result['SteadyStateValue']}")

        plt.plot(response.time, response.outputs)
        plt.show()
        
    def __call__(self, mutation_rate=0.5, *args: Any, **kwds: Any) -> Any:
        population = self.create_populasi()

        looping = True
        generation = 0

        while looping:

            parent1, parent2 = self.selection(population)

            child1, child2 = self.crossover(parent1, parent2)

            mutation1 = self.mutation(child1, mutation_rate)
            mutation2 = self.mutation(child2, mutation_rate)

            fitness_mutation1, gen1 = calculation_mutation(mutation1, self.num, self.den, self.n_var, self.n_bit, self.ra, self.rb)
            fitness_mutation2, gen2 = calculation_mutation(mutation2, self.num, self.den, self.n_var, self.n_bit, self.ra, self.rb)

            mutation1["gen"] = gen1
            mutation2["gen"] = gen2 
            mutation1["fitness"] = fitness_mutation1
            mutation2["fitness"] = fitness_mutation2

            population = self.regeneration([mutation1, mutation2], population)
            best, looping = self.termination(population)
            self.display_out(best, generation)

            generation += 1

        self.get_PID(self.num, self.den, best)
