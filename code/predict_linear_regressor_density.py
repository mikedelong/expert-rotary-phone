import logging
import random
import time

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

    intervals_count = 10
    delta = (max(xs) - min(xs)) / float(intervals_count)
    logger.debug('intervals_count : %d, delta = %.4f' % (intervals_count, delta))
    interval_starts = [min(xs) + index * delta for index in range(0, intervals_count + 1)]
    logger.debug('interval starts: %s' % interval_starts)

    result_x = list()
    result_y = list()
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


    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
