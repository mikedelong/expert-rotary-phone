import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

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

    # let's get a bunch of points
    real_size = 100
    synthetic_size = 200
    noise = 1.0
    xs = np.linspace(1, 10, real_size)
    # let's add a small noise term to get the y coordinates
    ys = xs + np.random.uniform(-noise / 2, noise / 2, size=real_size)
    model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
    X = pd.DataFrame.from_dict({'x': xs})
    model.fit(X, ys)

    zs = np.random.uniform(min(xs), max(xs), size=synthetic_size).reshape(-1, 1)
    predicted = model.predict(zs)
    score = model.score(X, ys)
    logger.debug('model score: %.4f' % score)
    out_noise = np.sqrt(1.0 - score)
    logger.debug(out_noise)

    plt.scatter(xs, ys, c='black')
    plt.scatter(zs, predicted, c='red')
    out_file = '../output/regressor_prediction.png'
    plt.savefig(out_file)
    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
