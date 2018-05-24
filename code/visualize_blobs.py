import logging
import time

from matplotlib.pyplot import savefig
from matplotlib.pyplot import subplots
from seaborn import kdeplot
from sklearn.datasets import make_blobs

start_time = time.time()

formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)
logger.debug('started')

nrows = 1
ncols = 2
figure, ([axes_00, axes_01]) = subplots(ncols=ncols, nrows=nrows, figsize=(5 * ncols, 5 * nrows))
random_state = 1

points = 1000

x1, y1 = make_blobs(points, n_features=2, centers=1, random_state=random_state)
axes_00.scatter(x1[:, 0], x1[:, 1], c='green', s=1)
kdeplot(x1[:, 0], x1[:, 1], ax=axes_01)

output_file = '../output/blobs_and_contours.png'
logger.debug('saving blob pictures to %s' % output_file)
savefig(output_file)

logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
console_handler.close()
logger.removeHandler(console_handler)
