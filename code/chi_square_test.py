import logging
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import subplots
from scipy.stats import chisquare
from scipy.stats import gaussian_kde

start_time = time.time()

formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)
logger.debug('started')

random_seed = 2
np.random.seed(random_seed)
sample_counts = [100, 500, 1000, 5000, 10000, 50000]
ncols = 2
nrows = len(sample_counts)
figure, axes = subplots(ncols=ncols, nrows=nrows, figsize=(3 * ncols, 3 * nrows))
for index, sample_count in enumerate(sample_counts):
    t0 = np.random.normal(0, 1, sample_count)
    t1 = np.random.normal(0, 1, sample_count)

    kde_samples = sample_count // 10

    axes[index, 0].hist(t0, density=True, color='blue')
    kde_t0 = gaussian_kde(t0)
    pts_t0 = np.linspace(min(t0), max(t0), kde_samples)
    axes[index, 0].plot(pts_t0, kde_t0(pts_t0), c='orange')
    title_0 = '{} samples'.format(sample_count)
    axes[index, 0].set_title(title_0)
    kde_t1 = gaussian_kde(t1)
    pts_t1 = np.linspace(min(t1), max(t1), kde_samples)
    axes[index, 1].hist(t1, density=True, color='blue')
    axes[index, 1].plot(pts_t1, kde_t1(pts_t1), c='orange')
    t3 = chisquare(t0, t1, ddof=0, axis=None)

    t4 = int(1000 * t3.pvalue)
    t5 = float(t4) / 1000
    title_1 = 'p-value: {}'.format(t5)
    axes[index, 1].set_title(title_1)

    logger.debug('samples: %d p-value: %.3f' % (sample_count, t3.pvalue))

plt.tight_layout()
plt.savefig('../output/chi_square_examples.png')
logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
console_handler.close()
logger.removeHandler(console_handler)
