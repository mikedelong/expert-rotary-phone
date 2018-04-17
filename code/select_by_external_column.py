import pandas as pd

column = ['A', 'B', 'A', 'A']

t0 = pd.DataFrame.from_dict({'A': [1.0, 0.0], 'B': [1.0, 1.0], 'C': [0.0, 0.0], 'D': [1.0, 1.0]})

print(t0)
chooser = [item for index, item in enumerate(t0.columns.values) if column[index] == 'A']
print(t0[chooser])
