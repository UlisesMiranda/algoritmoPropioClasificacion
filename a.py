import pandas as pd

nombres_columnas = ['id', 'age', 'preescription', 'astigmatic', 'tear_rate', 'target']
df = pd.read_csv('lenses/lenses.data', header=None, sep='\s+', names=nombres_columnas, index_col=False)

df = df.drop(df.columns[0], axis=1)

print(df)

df.to_csv('lenses.csv', index=False)