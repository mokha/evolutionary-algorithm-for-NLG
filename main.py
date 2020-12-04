from collections import defaultdict
import logging
from genetic_algorithm import GeneticAlgorithm
from creative_code import *
from mosestokenizer import *


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    tokenize = MosesTokenizer('en')
    detokenize = MosesDetokenizer('en')

    def print_output(output):
        for _r in output:
            logger.info("%s: %s", detokenize(_r), str(_r.fitness.values))
        logger.info(" ")

    population_size = 25
    max_iter = 5
    crossover_prob = 0.2
    mutation_prob = 0.8
    ga_weights = weights

    generator = GeneticAlgorithm(population_size=population_size, max_iter=max_iter, crossover_prob=crossover_prob,
                                 mutation_prob=mutation_prob, weights=ga_weights)

    text = 'Suffering from Success'
    tokens = tokenize(text)
    pop, _, hof = generator.run(tokens)

    print_output(hof)

    tokenize.close()
    detokenize.close()


if __name__ == '__main__':
    main()
