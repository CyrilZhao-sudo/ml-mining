import random
import numpy as np
from deap import base, creator, tools
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from src.utils import print_rank_lift, compute_auc_ks

def get_lift(prob, label, p=0.1):
    df = pd.DataFrame({"prob":prob, "label":label})
    df.sort_values(by=['prob'], axis=0, ascending=False, inplace=True, ignore_index=True)
    reject_rank, base_default_rate = int(p * len(df)), np.mean(df["label"])
    prob_threshold= df.loc[reject_rank-1, "prob"]
    reject_df = df.loc[df["prob"]>=prob_threshold, :]
    reject_bad_rate = np.mean(reject_df["label"])
    reject_bad_n = np.sum(reject_df["label"])
    reject_n = len(reject_df)
    return reject_bad_rate / base_default_rate

def get_auc(prob, label):
    auc = roc_auc_score(y_true=label, y_score=prob)
    return auc




def generate_weight(num):
    value = np.random.random(num)
    total_sum = np.sum(value)
    for v in value:
        yield v / total_sum


def cxTwoPoint(ind1, ind2):
    size = min(len(ind1), len(ind2))
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]

    total_ind1 = sum(ind1)
    total_ind2 = sum(ind2)
    for i, v in enumerate(ind1):
        ind1[i] = v / total_ind1
    for i, v in enumerate(ind2):
        ind2[i] = v / total_ind2
    return ind1, ind2


class SimpleGeneticAlgorithm(object):
    """
    Modified genetic algorithm, used to solve following constraint optimization problems

        min  f(w1, w2, w3, w4, w5, ... )

        s.t.   0<= w_i <=1, i=1, 2, 3, ...
               w_1 + w_2 + ... w_n = 1

               # lower bound and upper bound can be defined by user
               lower_bound <= w_i <= upper_bound, i=1, 2, 3, ...

    where f can be any objective functions, such as cross entropy, mse, etc.
    """

    def __init__(self,
                 x,
                 y,
                 population_size=1000,
                 generation=500,
                 cross_prob=0.3,
                 mutation_prob=0.4,
                 obj_fn='cross_entropy',
                 threshold=None,
                 early_stop=True,
                 verbose=False):
        """
        :param x: numpy array is required
        :param y: numpy array is required
        :param population_size: population size for GA algorithm, default 1000
        :param generation: number of generations, default 500
        :param cross_prob: the probability for cross event between two individuals
        :param mutation_prob: the probability for mutation
        :param obj_fn: objective functions used in the algorithm, currently support, default cross_entropy
            - mse: mean square error
            - cross_entropy: cross entropy function, same as objective function for logistic regression

        :param threshold: list type object, [lower_bound, upper_bound], contains lower bound and upper bound for the weight
        :param early_stop: default True, whether the algorithm stops early, if no improvement for 100 steps
        :param verbose: default False, whether to print the log information each step
        """

        self.x = x
        self.y = y
        self.population = population_size
        self.generation = generation
        self.cross_prob = cross_prob
        self.mutation_prob = mutation_prob
        self.threshold = threshold
        self.early_stop = early_stop
        self.verbose = verbose

        self.best_ind = None
        self.pop_last = None

        creator.create('GA', base.Fitness, weights=(-1,))
        creator.create('Individual', list, fitness=creator.GA)
        self.toolbox = base.Toolbox()

        self.toolbox.register('generate_weight', generate_weight, self.x.shape[1])
        self.toolbox.register(
            'individual',
            tools.initIterate,
            creator.Individual,
            self.toolbox.generate_weight
        )
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)

        self.obj_fn = obj_fn
        if obj_fn == 'cross_entropy':
            self.toolbox.register('evaluate', self.logistic_objective)
        elif obj_fn == 'mse':
            self.toolbox.register('evaluate', self.mse_objective)
        elif obj_fn == 'lift':
            self.toolbox.register('evaluate', self.lift_objective)
        elif obj_fn == 'ks':
            self.toolbox.register('evaluate', self.ks_objective)
        elif obj_fn == 'auc':
            self.toolbox.register('evaluate', self.auc_objective)
        else:
            raise NotImplementedError('More Objective Function in future')

        self.toolbox.register('mate', cxTwoPoint)
        self.toolbox.register('mutate', tools.mutShuffleIndexes, indpb=0.1)
        self.toolbox.register('select', tools.selTournament, tournsize=3)

    def meet_other_constraints(self, individual):
        """
        use to check whether each individual is between the user defined lower and upper bound if provided
        """

        if self.threshold is not None:
            if np.any(np.array(individual) > self.threshold[1]) | \
                    np.any(np.array(individual) < self.threshold[0]):
                return False
            else:
                return True
        else:
            return True

    def logistic_objective(self, individual):
        """
        cross entropy function, same as objective function in logistic regression
        """
        if not self.meet_other_constraints(individual):
            return np.iinfo(np.uint64).max,
        y_hat = np.matmul(self.x, individual)
        return -np.mean(self.y * np.log(y_hat) + (1 - self.y) * np.log(1 - y_hat)),

    def mse_objective(self, individual):
        """
        mean square error
        """
        if not self.meet_other_constraints(individual):
            return np.iinfo(np.uint64).max,
        y_hat = np.matmul(self.x, individual)
        return np.mean((self.y - y_hat) ** 2),

    def lift_objective(self, individual):
        if not self.meet_other_constraints(individual):
            return np.iinfo(np.uint64).max,
        y_hat = np.matmul(self.x, individual)
        lift  = get_lift(y_hat, self.y, p=0.032)
        return -lift,
    def auc_objective(self, individual):
        if not self.meet_other_constraints(individual):
            return np.iinfo(np.uint64).max,
        y_hat = np.matmul(self.x, individual)
        auc = roc_auc_score(self.y, y_hat)
        return -auc,
    def ks_objective(self, individual):
        if not self.meet_other_constraints(individual):
            return np.iinfo(np.uint64).max,
        y_hat = np.matmul(self.x, individual)
        fpr, tpr, _ = roc_curve(self.y, y_hat)
        ks = abs(max(fpr - tpr))
        return -ks,

    def run(self):
        """
        run the simplified GA algorithm
        """
        # initialize the population
        pop = self.toolbox.population(self.population)

        # evalute each individual in the population
        fitnesses = list(map(self.toolbox.evaluate, pop))

        # associate the value with each individual
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # 记录现在最小的目标函数的值(因为我们是最小化问题)
        previous_obj = None

        # used for early stop
        count = 0

        # loop for each generation
        for g in range(self.generation):
            if self.verbose:
                print("-- Generation %i --" % g)

            # select offsprings
            offspring = self.toolbox.select(pop, len(pop))
            # clone the current offsprings
            offspring = list(map(self.toolbox.clone, offspring))

            # between two individuals, check whether gene exchange will happen
            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                # if a random number is smaller than the cross_prob
                if random.random() < self.cross_prob:
                    self.toolbox.mate(child1, child2)

                    # delete the objective values for the children, since they are going to be reevaluated
                    del child1.fitness.values
                    del child2.fitness.values

            # for each individual
            for mutant in offspring:

                # if a random number is smaller than the mutation_prob
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(mutant)

                    # delete the objective values for the child, since it's going to be reevaluated
                    del mutant.fitness.values

            # select all the newly generated individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            # reevaluate the objective for newly generated individuals
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            # assign the value
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # update all the pupulations
            pop[:] = offspring

            # get all the objective functions
            fits = [ind.fitness.values[0] for ind in pop]

            # record the minumum
            current_obj = min(fits)

            if self.verbose:
                length = len(pop)
                mean = sum(fits) / length
                sum2 = sum(x * x for x in fits)
                std = abs(sum2 / length - mean ** 2) ** 0.5

                print("  Min %s" % min(fits))
                print("  Max %s" % max(fits))
                print("  Avg %s" % mean)
                print("  Std %s" % std)

            if previous_obj is None:
                previous_obj = current_obj
            elif previous_obj <= current_obj:
                # count none decreasing times
                count += 1
            else:
                # update the best objective values
                previous_obj = current_obj
                # previous_obj > current_obj:
                # reset decreasing times to 0
                count = 0

            if self.early_stop and count > 100:
                if self.verbose:
                    print('Early stop condition encountered!')
                break

        self.best_ind = tools.selBest(pop, 1)[0]
        self.pop_last = pop

        if self.verbose:
            print("-- End of (successful) evolution --")
            best_ind = tools.selBest(pop, 1)[0]
            print("Best individual is %s" % best_ind)
            print("Best objective function is %s" % best_ind.fitness.values)

if __name__ == '__main__':
    X = np.array([
        [0.5, 0.2, 0.4, 0.8],
        [0.3, 0.5, 0.1, 0.2],
        [0.4, 0.5, 0.2, 0.1],
        [0.9, 0.2, 0.8, 0.8]
    ])
    X1 = np.array([
        [500, 0.2, 0.4, 0.8],
        [300, 0.5, 0.1, 0.2],
        [400, 0.5, 0.2, 0.1],
        [900, 0.2, 0.8, 0.8]
    ])
    X2 = np.array([
        [500, 0.2, 0.4, 800],
        [300, 0.5, 0.1, 200],
        [400, 0.5, 0.2, 100],
        [900, 0.2, 0.8, 800]
    ])
    X3 = np.array([
        [500, 0.2, 0.4, 0.8],
        [0.3, 0.5, 0.1, 0.2],
        [0.4, 0.5, 0.2, 0.1],
        [0.9, 0.2, 0.8, 0.8]
    ])
    X4 = np.array([
        [1.0, 0.2, 0.4, 0.8],
        [0.3, 0.5, 0.1, 0.2],
        [0.4, 0.5, 0.2, 0.1],
        [0.9, 0.2, 0.8, 0.8]
    ])

    y = np.array([1, 0, 0, 1])
    print(X1)
    print(y)

    population = 500
    cross_prob = 0.5
    mutation_prob = 0.2
    generation = 100

    # threshold_max 用于限制权重最大值
    # threshold_min 用于限制权重最小值
    threshold_max = 0.33
    threshold_min = 0.05

    ga = SimpleGeneticAlgorithm(
        x=X,
        y=y,
        population_size=population,
        generation=1000,
        cross_prob=cross_prob,
        mutation_prob=mutation_prob,
        obj_fn='cross_entropy',
        threshold=[threshold_min, threshold_max],
        early_stop=True,
        verbose=True
    )

    ga.run()

    # best weights
    ga.best_ind

    # objective value
    ga.best_ind.fitness.values

#量纲不同处理

# result = {'test1':[], 'test2':[], 'test3':[], 'test4':[], 'test5':[]}
#
# data = [X, X1, X2, X3, X4]
# y = np.array([1, 0, 0, 1])
#
# for i, v in enumerate(data):
#     ga = SimpleGeneticAlgorithm(
#         x=v,
#         y=y,
#         population_size=population,
#         generation=1000,
#         cross_prob=cross_prob,
#         mutation_prob=mutation_prob,
#         obj_fn='cross_entropy',
#         threshold=[threshold_min, threshold_max],
#         early_stop=True,
#         verbose=True
#     )
#     ga.run()
#     res = ga.best_ind
#     key = 'test' + str(i+1)
#     result[key].extend(res)
#
# data_befor = np.array([0.708333333,0,0,1,0,0,0.25,0,0.4,0.1450777,0.105263158])
# data_after = np.array([0.695652174,0.5,0.076923077,0.235294118,0.235294118,0.25,0.2,0.181818182,0.153846154,0.126315789,0.116766467])
#
