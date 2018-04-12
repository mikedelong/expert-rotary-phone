import pandas as pd

df_left = pd.DataFrame({'A1': [1, 2, 3], 'B': [4, 5, 6]})
df_right = pd.DataFrame({'A2': [7, 8, 9], 'B': [10, 5, 6]})

join_result = df_left.join(df_right, on='B', rsuffix='_r')
print(join_result)
merge_result = df_left.merge(df_right, on='B')
print(merge_result)
