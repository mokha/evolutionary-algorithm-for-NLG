This code is meant as a starting point for implementing NLG solutions using genetic algorithms.

To run the sample code, you need the below requirements:
``pip install numpy deap mosestokenizer``

- `main.py`: The main script to execute the genetic algorithms. The input is handled here along with any parsing/tokenizations. Parameters to the genetic algorithm (e.g. population size, number of generations) are defined here.
- `genetic_algorithm.py`: The implementation of the evolutionary algorithm.
- `creative_code.py`: Contains the mutation, crossover and evaluation functions
  - weights: A tuple containing the weights (between -1 to +1) of each optimization dimension. A positive weight indicates maximizing the dimension and negative is for minimizing it.
  - mutate: This function is in charge of mutating a single individual (e.g. changing one token with a one). The function should return a tuple, so the comma after the individual is needed.
  - crossover: This function takes two individuals. It picks a point at random and swaps tokens (that are before the selected point) of both individuals.
  - evaluate_ind: A function that evaluates an individual. The function receives the current individual and the original individual that was set in `main.py`. The function must return the same number of scores as the number of dimensions set by `weights`.

The code could be extended to use additional information of the input (such as dependency parse-trees, themes and such). For that, the `genetic_algorithm.py` and `main.py` would need to be modified accordingly.

Methods that have employed evolutionary algorithms to generate creative language (in English and Finnish):
- Khalid Alnajjar and Hannu Hannu Toivonen (2020). [Computational Generation of Slogans](https://doi.org/10.1017/S1351324920000236). Natural Language Engineering.
- Mika Hämäläinen and Khalid Alnajjar (2019). [Let's FACE it. Finnish Poetry Generation with Aesthetics and Framing](https://www.aclweb.org/anthology/W19-8637/). In *the Proceedings of The 12th International Conference on Natural Language Generation*. pages 290-300.
  - GitHub: [FinMeter](https://github.com/mikahama/FinMeter)
