import logging
import time

import matplotlib.pyplot as plt
import numpy as np
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

nrows = 2
ncols = 3
figure, ([axes_00, axes_01, axes_02], [axes_10, axes_11, axes_12]) = plt.subplots(nrows=nrows, ncols=ncols,
                                                                                  figsize=(5 * nrows, 5 * ncols))

data0 = np.random.uniform(0, 1, size=100)
density0 = gaussian_kde(data0)
density0.covariance_factor = lambda: .25
density0._compute_covariance()
logger.debug(density0.dataset)
axes_00.plot(data0, 'r.', markersize=1)
xs0 = np.linspace(min(data0), max(data0), 100)
axes_01.plot(xs0, density0(xs0), 'b-')
axes_02.plot(density0.dataset[0], 'g-')

data1 = np.random.uniform(0, 1, size=100)
density1 = gaussian_kde(data1)
density1.covariance_factor = lambda: .25
density1._compute_covariance()
logger.debug(density1.dataset)
axes_10.plot(data1, 'r.', markersize=1)
xs1 = np.linspace(min(data1), max(data1), 100)
axes_11.plot(xs1, density1(xs1), 'b-')
axes_12.plot(density1.dataset[0], 'g-')

plt.show()

logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
console_handler.close()
logger.removeHandler(console_handler)
