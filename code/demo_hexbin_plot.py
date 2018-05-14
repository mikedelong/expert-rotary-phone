import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

    mean, cov = [0, 1], [(1, .5), (.5, 1)]
    x, y = np.random.multivariate_normal(mean, cov, 1000).T
    output_file = '../output/multivariate_normal_hexbin.png'
    with sns.axes_style("white"):
        sns.jointplot(x=x, y=y, kind="hex", color="k")
        logger.debug('saving plot to %s' % output_file)
        plt.savefig(output_file)

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
