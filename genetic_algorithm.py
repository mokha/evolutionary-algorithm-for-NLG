# -*- coding: utf-8 -*-
import numpy as np
import time
import logging
import sys
import deap.gp
import deap.benchmarks
from deap import creator, base, tools, algorithms
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence
import copy
from creative_code import crossover, mutate, evaluate_ind

logger = logging.getLogger(__name__)


class GeneticAlgorithm:
    def __init__(self, population_size=100, max_iter=5, crossover_prob=0.5, mutation_prob=0.5, weights=(1.0,)):
        logger.info('Initializing generator...')

        self.population_size = population_size
        self.max_iter = max_iter
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.weights = weights

        self.genetic_generator = GeneticAlgorithm.GeneticGenerator(self)
        logger.info('Generator initialized!')

    class GeneticGenerator:
        def __init__(self, generator):
            self.g = generator

        def set_parameters(self, text):
            self.tokens = copy.deepcopy(text)
            self.register_deap()

        def register_deap(self):
            self.creator = creator
            self.creator.create("Fitness", base.Fitness, weights=self.g.weights)
            self.creator.create("Individual", list, fitness=self.creator.Fitness)

            self.toolbox = base.Toolbox()
            self.toolbox.register("individual", self.individual)
            self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
            self.toolbox.register("evaluate", self.evaluate)
            self.toolbox.register("mate", crossover)
            self.toolbox.register("mutate", mutate)
            self.toolbox.register("select", tools.selNSGA2)

        def create_custom_obj(self, tokens):
            ind = self.creator.Individual(tokens)
            return ind

        def individual(self):
            tokens = copy.deepcopy(self.tokens)
            ind = self.create_custom_obj(tokens)
            return ind

        def evaluate(self, ind):
            e = evaluate_ind(ind, self.tokens)
            return e

        def generate(self):
            start = time.time()
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("min", np.min, axis=0)
            stats.register("avg", np.mean, axis=0)
            stats.register("max", np.max, axis=0)
            stats.register("std", np.std, axis=0)

            logbook = tools.Logbook()
            logbook.header = "gen", "evals", "min", "std", "avg", "max"

            toolbox = self.toolbox
            mu = self.g.population_size
            lambda_ = self.g.population_size
            cxpb = self.g.crossover_prob
            mutpb = self.g.mutation_prob
            ngen = self.g.max_iter
            halloffame = tools.ParetoFront()

            populations = []
            population = self.toolbox.population(n=self.g.population_size)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            populations.append(population)

            if halloffame is not None:
                halloffame.update(population)

            record = stats.compile(population)
            logbook.record(gen=0, nevals=len(population), **record)
            logger.info(logbook.stream)

            # Begin the generational process
            for gen in range(1, ngen + 1):
                # Vary the population
                offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Update the hall of fame with the generated individuals
                if halloffame is not None:
                    halloffame.update(offspring)

                # Select the next generation population
                population[:] = toolbox.select(population + offspring, mu)
                populations.append(population)

                # Update the statistics with the new population
                record = stats.compile(population) if stats is not None else {}
                logbook.record(gen=gen, nevals=len(population), **record)
                logger.info(logbook.stream)

            end = time.time()
            elapsed = end - start
            logger.info("Time for entire generation: %.4f", elapsed)

            return population, logbook, halloffame

    def run(self, tokens):
        self.genetic_generator.set_parameters(tokens)
        return self.genetic_generator.generate()
