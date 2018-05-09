import json
import logging
import time
from zipfile import ZipFile

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

with open('./sensor_time_delta_index.json', 'r', encoding='utf-8') as settings_fp:
    settings = json.load(settings_fp)

logger.debug(settings)
input_folder = settings['input_folder']
input_file = settings['input_file']
full_input_file = input_folder + input_file
logger.debug('we are reading our data from %s' % full_input_file)

n_jobs = 1
if 'n_jobs' in settings.keys():
    n_jobs = settings['n_jobs']
if n_jobs == 1:
    logger.debug('our model will run serial')
else:
    logger.debug('our model will run using %d jobs' % n_jobs)

nrows = None
if 'nrows' in settings.keys():
    nrows = settings['nrows']
    if nrows < 1:
        nrows = None

random_state = 1
if 'random_state' in settings.keys():
    random_state = settings['random_state']

zip_file = ZipFile(full_input_file)
logger.debug(zip_file.filelist)
columns = settings['named_columns']
logger.debug(columns)
logger.debug([text_file.filename for text_file in zip_file.infolist()])
dfs = {
    text_file.filename: pd.read_csv(zip_file.open(text_file.filename), names=columns,
                                    nrows=nrows,
                                    delim_whitespace=True,
                                    skiprows=1) for text_file in zip_file.infolist() if
    text_file.filename.endswith('.txt')}

output_compression = None
if 'output_compression' in settings.keys():
    output_compression = settings['output_compression']
output_folder = settings['output_folder']
for key in dfs.keys():
    data = dfs[key]
    data['seconds'] = pd.TimedeltaIndex(data=data['seconds'], unit='s')
    dfs[key] = data
    logger.debug('key: %s \n%s' % (key, data.head(10)))
    # save the result as CSV
    output_filename = key.replace('.txt', '.csv')
    full_output_file = output_folder + output_filename
    if output_compression is not None:
        if output_compression == 'gzip':
            full_output_file += '.gz'
        logger.debug('writing resulting CSV to %s using %s compression' % (full_output_file, output_compression))
    else:
        logger.debug('writing resulting CSV to %s' % full_output_file)
    data.to_csv(full_output_file, compression=output_compression)

logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
console_handler.close()
logger.removeHandler(console_handler)
