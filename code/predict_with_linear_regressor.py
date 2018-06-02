import logging
import math
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def solve_quadratic(a, b, c):
    discriminant = b ** 2 - 4 * a * c  # discriminant

    if discriminant < 0:
        raise ValueError('No solutions')
    elif discriminant == 0:
        x1 = -b / (2 * a)
        return x1, x1
    else:  # if d > 0
        d_sqrt = math.sqrt(discriminant)
        x1 = (-b + d_sqrt) / (2 * a)
        x2 = (-b - d_sqrt) / (2 * a)
        return x1, x2

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

    random_seed = 2
    np.random.seed(random_seed)
    # let's get a bunch of points
    real_size = 100
    synthetic_size = 200
    noise = 1.0
    # these are evenly spaced; let's make them non-uniform
    xs = np.random.uniform(1, 10, size=real_size)
    # let's add a small noise term to get the y coordinates
    real_noise_term = np.random.uniform(-noise / 2.0, noise / 2.0, size=real_size)
    ys = xs + real_noise_term
    model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
    X = np.array(xs.reshape(-1, 1))
    model.fit(X, ys)
    zs = np.random.uniform(min(xs), max(xs), size=synthetic_size).reshape(-1, 1)

    synthetic_noise_term = np.random.uniform(-noise / 3.0, 2.0 * noise / 3.0, size=synthetic_size)
    predicted = model.predict(zs) + synthetic_noise_term
    score = model.score(X, ys)
    logger.debug('model score: %.6f' % score)
    logger.debug('model coefficient and intercept: %.4f %.4f' % (model.coef_, model.intercept_))

    # now let's fit a second model to the predicted data
    post_model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
    post_X = np.array(zs)
    post_model.fit(post_X, predicted)
    logger.debug('post score: %.6f' % post_model.score(post_X, predicted))

    figure = plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, c='black', marker='o', s=3)
    plt.scatter(zs, predicted, c='red', marker='o', s=3)
    out_file = '../output/regressor_prediction.png'
    logger.debug('writing scatter plot to %s' % out_file)
    plt.savefig(out_file)
    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
