import datetime
import logging
import random
import time

gene_set = ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!.'
target = 'Hello World!'
target = 'the Quick brown Fox jumped Over the Lazy dog.'

start_time = time.time()
start_date_time = datetime.datetime.now()


def generate_parent(length):
    genes = []
    while len(genes) < length:
        sample_size = min(length - len(genes), len(gene_set))
        genes.extend(random.sample(gene_set, sample_size))
    return ''.join(genes)


def get_fitness(guess):
    return sum(1 for expected, actual in zip(target, guess)
               if expected == actual)


def mutate(parent):
    index = random.randrange(0, len(parent))
    child_genes = list(parent)
    new_gene, alternate = random.sample(gene_set, 2)
    child_genes[index] = alternate if new_gene == child_genes[index] else new_gene
    return ''.join(child_genes)


def display(guess):
    time_diff = datetime.datetime.now() - start_date_time
    fitness = get_fitness(guess)
    logger.debug('{0}\t{1}\t{2}'.format(guess, fitness, str(time_diff)))


formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)
logger.debug('started')

random.seed(1)
startTime = datetime.datetime.now()
bestParent = generate_parent(len(target))
bestFitness = get_fitness(bestParent)
display(bestParent)

while True:
    child = mutate(bestParent)
    childFitness = get_fitness(child)

    if bestFitness >= childFitness:
        continue
    display(child)
    if childFitness >= len(bestParent):
        break
    bestFitness = childFitness
    bestParent = child

logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
console_handler.close()
logger.removeHandler(console_handler)
