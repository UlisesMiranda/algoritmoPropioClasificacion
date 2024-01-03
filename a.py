import pandas as pd

nombres_columnas = ['target', 'c-shape', 'c-surface', 'c-color', 'bruises', 'odor', 'g-attachment', 'g-spacing', 'g-size', 'g-color', 's-shape', 's-root', 'surface-a-r', 'surface-b-r', 'color-a-r', 'color-b-r', 'v-type', 'v-color', 'r-number', 'r-type', 's-p-color', 'population', 'habitat']
df = pd.read_csv('mushroom/agaricus-lepiota.data', header=None, sep=',', names=nombres_columnas, index_col=False)

target_values = df['target']
df = df.drop(df.columns[0], axis=1)

df['target'] = target_values
print(df)

df.to_csv('mushroom.csv', index=False)