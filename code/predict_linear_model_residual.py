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
    real_size = 100
    synthetic_size = 100
    xs_max = 10
    noise = 2.0
    xs = np.random.uniform(1, xs_max, size=real_size)
    real_noise_term = np.random.uniform(-noise / 2.0, noise / 2.0, size=real_size)
    ys = xs + real_noise_term

    model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
    X = np.array(xs.reshape(-1, 1))
    model.fit(X, ys)
    zs = np.random.uniform(min(xs), max(xs), size=synthetic_size).reshape(-1, 1)
    predicted = model.predict(X=zs)
    logger.debug(
        'model score: %.6f coefficient: %.4f intercept: %.4f' % (model.score(X, ys), model.coef_, model.intercept_))

    # calculate the residuals
    residuals = [ys[index] - model.coef_ * xs[index] - model.intercept_ for index in range(real_size)]
    min_residual = min(residuals)
    max_residuals = max(residuals)

    x_min = np.min(xs)
    x_max = np.max(xs)
    x_synthetic = np.random.uniform(x_min, x_max, synthetic_size)
    synthetic_noise = np.random.uniform(min_residual, max_residuals, synthetic_size)
    y_synthetic = model.coef_ * x_synthetic + model.intercept_ + synthetic_noise

    # now let's fit a second model to the predicted data
    post_model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
    post_X = np.array(x_synthetic).reshape(-1, 1)
    post_model.fit(post_X, y_synthetic)
    logger.debug('synthetic model/synthetic data score: %.6f coefficient: %.4f intercept: %.4f' % (
        post_model.score(post_X, y_synthetic), post_model.coef_, post_model.intercept_))
    logger.debug('synthetic model/real data score: %.6f' % post_model.score(X, ys))
    logger.debug('real model/synthetic data score: %.6f' % model.score(post_X, y_synthetic))

    figure = plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, c='black', marker='o', s=2)
    plt.scatter(x_synthetic, y_synthetic, c='red', marker='o', s=2)
    out_file = '../output/regressor_residual_prediction.png'
    logger.debug('writing scatter plot to %s' % out_file)
    plt.savefig(out_file)

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
