import logging
import time

import tensorflow as tf

start_time = time.time()

formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)
logger.debug('started')

tf.logging.set_verbosity(tf.logging.DEBUG)

a = tf.constant(5, name="a")
b = tf.constant(15, name="b")
c = tf.add(a, b, name="c")

logger.debug("Value of c before running tensor: %s" % c)
session = tf.Session()
output = session.run(c)
logger.debug("Value of c after running graph: %s" % output)
session.close()

logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
console_handler.close()
logger.removeHandler(console_handler)
