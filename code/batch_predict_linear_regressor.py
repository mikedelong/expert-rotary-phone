import logging
import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def batch(iterable, n=1):
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx:min(ndx + n, length)]


def solve_quadratic(a, b, c):
    discriminant = b * b - 4 * a * c  # discriminant

    if discriminant < 0:
        raise ValueError('No solutions')
    elif discriminant == 0:
        result_1 = -b / (2 * a)
        return result_1, result_1
    else:  # if d > 0
        d_sqrt = math.sqrt(discriminant)
        result_1 = (-b + d_sqrt) / (2 * a)
        result_2 = (-b - d_sqrt) / (2 * a)
        return result_1, result_2


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
    random.seed(random_seed)
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
    predicted = model.predict(X=zs)
    score = model.score(X, ys)
    logger.debug('model score: %.6f' % score)
    logger.debug('model coefficient and intercept: %.4f %.4f' % (model.coef_, model.intercept_))

    real_batch_size = 5
    synthetic_batch_size = 10
    real_starts = range(0, real_size, real_batch_size)
    synthetic_starts = range(0, synthetic_size, synthetic_batch_size)

    synthetic_index = 0
    result = list()
    for index, real_start in enumerate(real_starts):
        batch_xs = xs[real_start:real_start + real_batch_size]
        batch_ys = ys[real_start:real_start + real_batch_size]
        batch_real_model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
        batch_Xs = np.array(batch_xs.reshape(-1, 1))
        batch_real_model.fit(batch_Xs, batch_ys)
        synthetic_start = synthetic_starts[index]
        batch_zs = zs[synthetic_start: synthetic_start + synthetic_batch_size]
        batch_predictions = batch_real_model.predict(X=batch_zs)
        batch_score = batch_real_model.score(batch_Xs, batch_ys)
        batch_prediction_mean = np.mean(batch_predictions)
        for z in batch_zs:
            f_i = model.coef_ * z + model.intercept_
            a_coef = batch_score
            b_coef = -2.0 * batch_prediction_mean * (batch_score - 1.0) - 2.0 * f_i
            c_coef = (batch_score - 1.0) * batch_prediction_mean * batch_prediction_mean + f_i * f_i
            x1, x2 = solve_quadratic(a_coef, b_coef, c_coef)
            y_i = random.choice([x1, x2])
            result.append(y_i)

    # now let's fit a second model to the predicted data
    post_model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
    post_X = np.array(zs)
    post_model.fit(post_X, result)
    logger.debug('post score: %.6f' % post_model.score(post_X, result))

    figure = plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, c='black', marker='o', s=3)
    plt.scatter(zs, result, c='red', marker='o', s=3)
    out_file = '../output/regressor_batch_prediction.png'
    logger.debug('writing scatter plot to %s' % out_file)
    plt.savefig(out_file)
    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
