import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

start_time = time.time()

formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)
logger.debug('started')

np.random.seed([3, 1415])
df = pd.DataFrame(dict(
    Name='matt joe adam farley'.split() * 100,
    Seconds=np.random.randint(4000, 5000, 400, np.int)
))

df['Zscore'] = df.groupby('Name').Seconds.apply(lambda x: x.div(x.mean()))

df.groupby('Name').Zscore.plot.kde()
output_folder = '../output/'
output_file = 'example_kde.png'
full_output_file = output_folder + output_file
plt.savefig(full_output_file)

logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
console_handler.close()
logger.removeHandler(console_handler)
