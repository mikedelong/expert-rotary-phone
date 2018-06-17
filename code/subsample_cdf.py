# https://matplotlib.org/examples/statistics/histogram_demo_cumulative.html
import logging

import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib import mlab

if __name__ == '__main__':
    start_time = time.time()

    formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    console_handler.setLevel(logging.DEBUG)
    logger.debug('started')

    random_seed = 1
    np.random.seed(random_seed)

    mu = 200
    sigma = 25
    n_bins = 50
    size = 1000
    x = np.random.normal(mu, sigma, size=size)

    x_shuffle = x.copy()
    np.random.shuffle(x_shuffle)
    x1 = x_shuffle[:size // 2]
    x2 = x_shuffle[size // 2:]
    fig, (ax, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 12))

    # plot the cumulative histogram
    n, bins, patches = ax.hist(x, n_bins, density=True, histtype='step', cumulative=True, label='Empirical')
    n1, bins1, patches1 = ax1.hist(x1, n_bins, density=True, histtype='step', cumulative=True, label='Empirical')
    n2, bins2, patches2 = ax2.hist(x2, n_bins, density=True, histtype='step', cumulative=True, label='Empirical')

    # Add a line showing the expected distribution.
    y = mlab.normpdf(bins, mu, sigma).cumsum()
    y /= y[-1]

    ax.plot(bins, y, 'k--', linewidth=1.5, label='Theoretical')

    # tidy up the figure
    ax.grid(True)
    ax1.grid(True)
    ax2.grid(True)
    ax.legend(loc='right')
    ax1.legend(loc='right')
    ax2.legend(loc='right')

    output_file = '../output/subsample_cdf.png'
    logger.debug('writing plot file to %s' % output_file)

    plt.savefig(output_file)

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
