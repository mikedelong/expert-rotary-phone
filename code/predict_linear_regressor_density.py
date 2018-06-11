import logging
import random
import time

import matplotlib.pyplot as plt
import numpy as np
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

    # random seeds
    random_seed = 2
    np.random.seed(random_seed)
    random.seed(random_seed)

    # generate random data using a nonuniform line plus a noise term
    real_size = 1000
    synthetic_size = 1000
    noise = 2.0
    xs = np.random.uniform(1, 100, size=real_size)
    real_noise_term = np.random.uniform(-noise / 2.0, noise / 2.0, size=real_size)
    ys = xs + real_noise_term

    model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
    X = np.array(xs.reshape(-1, 1))
    model.fit(X, ys)
    zs = np.random.uniform(min(xs), max(xs), size=synthetic_size).reshape(-1, 1)
    predicted = model.predict(X=zs)
    logger.debug(
        'model score: %.6f coefficient: %.4f intercept: %.4f' % (model.score(X, ys), model.coef_, model.intercept_))

    # todo choose the interval count more carefully
    intervals_count = int(xs.max()) - int(xs.min())
    delta = (max(xs) - min(xs)) / float(intervals_count)
    logger.debug('intervals_count : %d, delta = %.4f' % (intervals_count, delta))
    interval_starts = [min(xs) + index * delta for index in range(0, intervals_count + 1)]

    result_x = list()
    result_y = list()
    values_generated = 0
    for index, interval_start in enumerate(interval_starts[:-1]):
        lower = interval_start
        upper = interval_starts[index + 1]
        count = 0
        y_min = max(ys)
        y_max = min(ys)
        for jndex, value in enumerate(xs):
            if lower <= value < upper:
                count += 1
                y_value = ys[jndex]
                y_max = max(y_max, y_value)
                y_min = min(y_min, y_value)
        values_to_generate = count * synthetic_size // real_size
        for _ in range(values_to_generate):
            result_x.append(np.random.uniform(lower, upper))
            result_y.append(np.random.uniform(y_min, y_max))
        values_generated += values_to_generate

    logger.debug('values generated: %d actual %d expected' % (values_generated, synthetic_size))
    residual_count = synthetic_size - values_generated
    if residual_count > 0:
        for index in range(residual_count):
            # pick a random interval to top up
            interval = np.random.randint(0, intervals_count)
            lower = interval_starts[interval]
            upper = interval_starts[interval + 1]
            y_min = max(ys)
            y_max = min(ys)
            for jndex, value in enumerate(xs):
                if lower <= value < upper:
                    y_value = ys[jndex]
                    y_max = max(y_max, y_value)
                    y_min = min(y_min, y_value)
            result_x.append(np.random.uniform(lower, upper))
            result_y.append(np.random.uniform(y_min, y_max))
            values_generated += 1

    # now let's fit a second model to the predicted data
    post_model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
    post_X = np.array(result_x).reshape(-1, 1)
    post_model.fit(post_X, result_y)
    logger.debug('synthetic model/synthetic data score: %.6f coefficient: %.4f intercept: %.4f' % (
        post_model.score(post_X, result_y), post_model.coef_, post_model.intercept_))
    logger.debug('synthetic model/real data score: %.6f' % post_model.score(X, ys))
    logger.debug('real model/synthetic data score: %.6f' % model.score(post_X, result_y))

    figure = plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, c='black', marker='o', s=2)
    plt.scatter(result_x, result_y, c='red', marker='o', s=2)
    out_file = '../output/regressor_density_prediction.png'
    logger.debug('writing scatter plot to %s' % out_file)
    plt.savefig(out_file)

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
