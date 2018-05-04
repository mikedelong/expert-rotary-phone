import logging
import time

import pandas as pd

start_time = time.time()

if __name__ == '__main__':
    formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    console_handler.setLevel(logging.DEBUG)
    logger.debug('started')

    index_0 = pd.DatetimeIndex(['2018-05-04', '2018-05-05'], freq='1D')
    logger.debug(index_0)
    index_1 = pd.DatetimeIndex(start='2018-05-01', end='2018-05-05', freq='D')
    logger.debug(index_1)
    index_2 = index_1.to_period(freq='1D')
    logger.debug(index_2)

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
