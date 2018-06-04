import logging
import time

from predict_with_linear_regressor import solve_quadratic

if __name__ == '__main__':
    start_time = time.time()

    formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    console_handler.setLevel(logging.DEBUG)
    logger.debug('started')

    a_coef = 1.0
    b_coef = -3.0
    c_coef = 2.0
    x1, x2 = solve_quadratic(a=a_coef, b=b_coef, c=c_coef)

    logger.debug(x1)
    logger.debug(x2)
    solutions = [x1, x2]
    assert (1.0 in solutions)
    assert (2.0 in solutions)

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
