import json
import logging
import random
import string
import time
from os.path import basename
from os.path import exists

import tensorflow as tf

start_time = time.time()

alphabet = string.ascii_lowercase
suffix = '.py'
do_tensorflow = False
if __name__ == '__main__':
    formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    console_handler.setLevel(logging.DEBUG)
    logger.debug('started')
    logger.debug('alphabet: [%s]' % alphabet)

    settings_file = './read_and_write.json'
    with open(settings_file, 'r') as settings_fp:
        settings = json.load(settings_fp)
    random_seed = settings['random_seed']
    random.seed(random_seed)

    base_file_name = basename(__file__)
    logger.debug('our source file is %s' % base_file_name)
    with open(base_file_name, 'r') as base_file_fp:
        content = base_file_fp.read()

    short_file_name = base_file_name.replace(suffix, '')
    new_short_file_name = short_file_name
    done = False
    new_file_name = short_file_name
    while not done:
        # pick a location at random
        location = random.randint(0, len(short_file_name))
        logger.debug('location: %d' % location)
        letter = random.choice(alphabet)
        logger.debug('letter: %s' % letter)
        new_short_file_name = short_file_name[0:location] + letter + short_file_name[location + 1:]
        new_file_name = new_short_file_name + suffix
        logger.debug(new_file_name)
        if not exists(new_file_name):
            done = True

    content = content.replace(short_file_name, new_short_file_name)
    with open(new_file_name, 'w') as output_fp:
        output_fp.write(content)

    # still need to write out the new settings
    new_settings = settings.copy()
    new_settings['random_seed'] = random.randint(0, 100)
    new_settings_file = settings_file.replace(short_file_name, new_short_file_name)
    logger.debug('new settings file name is %s' % new_settings_file)
    with open(new_settings_file, 'w') as output_fp:
        json.dump(new_settings, output_fp)

    if do_tensorflow:
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
