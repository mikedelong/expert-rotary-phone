import json
import logging
import time

import matplotlib.pyplot as plt
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

with open('./visualize_sensor_time_series.json', 'r', encoding='utf-8') as settings_fp:
    settings = json.load(settings_fp)

logger.debug(settings)
input_folder = settings['input_folder']
input_file = settings['input_file']
nrows = settings['nrows']
if nrows == -1:
    nrows = None
output_folder = settings['output_folder']
time_column = settings['time_column']
full_input_file = input_folder + input_file
logger.debug('we are reading our data from %s' % full_input_file)

compression = None
if input_file.endswith('.gz'):
    compression = 'gzip'

data = pd.read_csv(full_input_file, compression=compression, nrows=nrows)
logger.debug('our data frame is %d x %d' % data.shape)
logger.debug(data.columns)

for column in data.columns:
    if column.startswith('Unnamed'):
        logger.debug('dropping column %s from data' % column)
        data.drop([column], axis=1, inplace=True)

# we know we have 18 columns to visualize
nrows = 6
ncols = 3
figure, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 8))

for index, column in enumerate(data.columns[1:]):
    irows = index // ncols
    icols = index % ncols
    axis = axes[irows, icols]
    data[[time_column, column]].plot(ax=axis)
    axis.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

plt.tight_layout()
if input_file.endswith('.gz'):
    output_file = input_file.replace('.gz', '')

output_file = output_file.replace('.csv', '.png')
full_output_file = output_folder + output_file
logger.debug('saving resulting figure to %s' % full_output_file)
plt.savefig(full_output_file)

logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
console_handler.close()
logger.removeHandler(console_handler)
