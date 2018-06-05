import logging
import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def solve_quadratic(a, b, c):
    discriminant = b ** 2 - 4 * a * c  # discriminant

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

    for index, item in enumerate(zs):
        prediction = model.coef_ * item + model.intercept_
        error = abs(prediction - predicted[index])
        if error > 0.0:
            logger.warning('linear model sanity check failed at step %d with error %.4f' % (index, error))

    score = model.score(X, ys)
    logger.debug('model score: %.6f' % score)
    logger.debug('model coefficient and intercept: %.4f %.4f' % (model.coef_, model.intercept_))

    variances = np.random.uniform(0.0, score / real_size, synthetic_size)
    y_mean = np.mean(predicted)
    logger.debug('y_mean: %.4f' % y_mean)
    variance = variances[0]
    a_coef = variance - 2.0
    b_coef = (variance - 1.0) * (-2.0 * y_mean) - 2.0 * predicted[0]
    c_coef = (variance - 1.0) * y_mean * y_mean + predicted[0] * predicted[0]
    logger.debug('quadratic coefficients: %.4f %.4f %.4f, variance: %.4f' % (a_coef, b_coef, c_coef, variance))
    x1, x2 = solve_quadratic(a_coef, b_coef, c_coef)
    logger.debug('quadratic solutions: %.4f %.4f predicted: %.4f' % (x1, x2, predicted[0]))

    # choose between x1 and x2 randomly
    y_0 = random.choice([x1, x2])

    # the numerator terms are the (y_i - f_i)^2 iterates
    numerator_terms = list()
    numerator_0 = y_0 - predicted[0]
    numerator_terms.append(numerator_0 * numerator_0)

    # the denominator terms are the (y_i - y_mean)^2 iterates
    denominator_terms = list()
    denominator_0 = (y_0 - y_mean)
    denominator_terms.append(denominator_0 * denominator_0)

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
