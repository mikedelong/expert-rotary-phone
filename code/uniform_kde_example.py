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
                                                                                  figsize=(5 * ncols, 5 * nrows))

points = 100
samples = 33
data0 = np.random.uniform(0, 1, size=points)
density0 = gaussian_kde(data0)
density0.covariance_factor = lambda: .25
density0._compute_covariance()
axes_00.plot(data0, 'r.', markersize=1)
axes_00.set_xlabel('data')
xs01 = np.linspace(min(data0), max(data0), points)
axes_01.plot(xs01, density0(xs01), 'b-')
axes_01.set_xlabel('KDE {} points'.format(points))
xs02 = np.linspace(min(data0), max(data0), samples)
axes_02.plot(xs02, density0(xs02), 'g-')
axes_02.set_xlabel('KDE {} points'.format(samples))

data1 = np.random.uniform(0, 1, size=points)
density1 = gaussian_kde(data1)
density1.covariance_factor = lambda: .25
density1._compute_covariance()
axes_10.plot(data1, 'r.', markersize=1)
axes_10.set_xlabel('data')
xs11 = np.linspace(min(data1), max(data1), points)
axes_11.plot(xs11, density1(xs11), 'b-')
axes_11.set_xlabel('KDE {} points'.format(points))
xs12 = np.linspace(min(data1), max(data1), samples)
axes_12.plot(xs12, density1(xs12), 'g-')
axes_12.set_xlabel('KDE {} points'.format(samples))

output_file = '../output/uniform_kde_samples.png'
logger.debug('saving view to %s' % output_file)
plt.savefig(output_file)
plt.close()

del figure
del axes_00
del axes_01
del axes_02
del axes_10
del axes_11
del axes_12

nrows = 2
ncols = 2
figure, ([axes_00, axes_01], [axes_10, axes_11]) = plt.subplots(nrows=nrows, ncols=ncols,
                                                                figsize=(4 * ncols, 4 * nrows))

axes_00.plot(data0, 'r.', markersize=1)
axes_00.set_xlabel('data set 0')
axes_01.plot(data1, 'b.', markersize=1)
axes_01.set_xlabel('data set 1')
axes_10.plot(xs01, density0(xs01), 'r-')
axes_10.plot(xs11, density1(xs11), 'b-')
axes_10.set_xlabel('KDE {} points'.format(points))
axes_11.plot(xs02, density0(xs02), 'r-')
axes_11.plot(xs12, density1(xs12), 'b-')
axes_11.set_xlabel('KDE {} points'.format(samples))
output_file = '../output/uniform_kde_overlapping.png'
logger.debug('saving view to %s' % output_file)
plt.savefig(output_file)

logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
console_handler.close()
logger.removeHandler(console_handler)
