# -*- coding: utf-8 -*-
import random
import copy
from deap.tools import cxOnePoint

weights = (+1.0, +1.0, +1.0, -1.0,)  # +1 for maximizing, -1 for minimizing

def mutate(ind):
    return ind,

def crossover(ind1, ind2):
    return cxOnePoint(ind1, ind2)

def evaluate_ind(ind, original_ind):
    ind = copy.deepcopy(ind)
    return (random.random(), random.random(), random.random(), random.random(),)
