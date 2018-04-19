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

with open('./read_zip_settings.json', 'rb') as settings_fp:
    settings = json.load(settings_fp)

logger.debug(settings)
input_folder = settings['input_folder']
input_file = settings['input_file']
full_input_file = input_folder + input_file
logger.debug('we are reading our data from %s' % full_input_file)

zip_file = ZipFile(full_input_file)
logger.debug(zip_file.filelist)
columns = settings['named_columns']
columns.extend(['s' + str(i) for i in range(16)])
logger.debug(columns)
logger.debug([text_file.filename for text_file in zip_file.infolist()])
dfs = {
text_file.filename: pd.read_csv(zip_file.open(text_file.filename), names=columns, nrows=10, delim_whitespace=True,
                                skiprows=1) for text_file in zip_file.infolist() if
text_file.filename.endswith('.txt')}

logger.debug(dfs.keys())
for key in dfs.keys():
    logger.debug(dfs[key].head())

logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
console_handler.close()
logger.removeHandler(console_handler)
