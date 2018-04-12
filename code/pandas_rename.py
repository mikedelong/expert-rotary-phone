import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df.columns.values)
df.rename(index=str, columns={'A': 'a', 'B': 'c'})
print(df.columns.values)
df = df.rename(index=str, columns={'A': 'a', 'B': 'c'})
print(df.columns.values)
df.rename(index=str, columns={'a': 'A'}, inplace=True)
print(df.columns.values)
