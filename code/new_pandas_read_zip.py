import json
import logging
import time
from zipfile import ZipFile

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz

start_time = time.time()

formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)
logger.debug('started')

# todo fix this
with open('./new_read_zip_settings.json', 'rb') as settings_fp:
    t = settings_fp.read()
    settings = json.loads(t.decode('utf-8'))

logger.debug(settings)
input_folder = settings['input_folder']
input_file = settings['input_file']
full_input_file = input_folder + input_file
logger.debug('we are reading our data from %s' % full_input_file)
nrows = None
if 'nrows' in settings.keys():
    nrows = settings['nrows']
    if nrows < 1:
        nrows = None
visualize_decision_tree = False
if 'visualize_decision_tree' in settings.keys():
    visualize_decision_tree = settings['visualize_decision_tree']

zip_file = ZipFile(full_input_file)
logger.debug(zip_file.filelist)
columns = settings['named_columns']
columns.extend(['s' + str(i) for i in range(16)])
logger.debug(columns)
logger.debug([text_file.filename for text_file in zip_file.infolist()])
dfs = {
    text_file.filename: pd.read_csv(zip_file.open(text_file.filename), names=columns,
                                    nrows=nrows,
                                    delim_whitespace=True,
                                    skiprows=1) for text_file in zip_file.infolist() if
    text_file.filename.endswith('.txt')}

key = list(dfs.keys())[1]
data = dfs[key]
training_columns = settings['training_columns']
target_columns = settings['target_columns']
X = data[training_columns]
logger.debug('our data is %d x %d' % X.shape)

# todo make these settings
test_size = 0.99
n_jobs = 10
random_state = 1

for target_column in target_columns:
    y = data[target_column]

    n_estimators = n_jobs  # note that we are pegging these
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logger.debug('our training features are %s' % X_train.columns)

    linear_model = LinearRegression(n_jobs=n_jobs)
    linear_model.fit(X=X_train, y=y_train)
    linear_model_score = linear_model.score(X=X_test, y=y_test)
    logger.debug('linear model: for test size %.3f we have accuracy %.3f' % (test_size, linear_model_score))

    decision_tree_model = DecisionTreeRegressor(random_state=random_state)
    decision_tree_model.fit(X=X_train, y=y_train)
    decision_tree_score = decision_tree_model.score(X=X_test, y=y_test)
    logger.debug('decision tree: for test size %.3f we have accuracy %.3f' % (test_size, decision_tree_score))
    if visualize_decision_tree:
        out_file = target_column + '_decision_tree.dot'
        export_graphviz(decision_tree_model, feature_names=X_train.columns, filled=True, rounded=True,
                        out_file=out_file)

    random_forest_model = RandomForestRegressor(n_jobs=n_jobs, n_estimators=n_estimators)
    random_forest_model.fit(X=X_train, y=y_train)
    random_forest_score = random_forest_model.score(X=X_test, y=y_test)
    logger.debug('random forests: for test size %.3f we have accuracy %.3f' % (test_size, random_forest_score))

logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
console_handler.close()
logger.removeHandler(console_handler)