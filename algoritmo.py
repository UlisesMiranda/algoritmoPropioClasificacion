import pandas as pd
from graphviz import Digraph
from sklearn.metrics import accuracy_score
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split

def manage_df(df: pd.DataFrame, target_column):
    label_encoder = LabelEncoder()
    df[target_column] = label_encoder.fit_transform(df[target_column])

    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    return X, y

class RedMarkovClasificacionCategorica:
    def __init__(self):
        self.matriz_transicion = None
        self.clases = None

    def fit(self, X_train, y_train):
        self.clases = sorted(y_train.unique())
        
        # Contadores para contar las transiciones de características a clase
        feature_counts = {}
        for feature in X_train.columns:
            feature_counts[feature] = {clase: {} for clase in self.clases}

        # Calcular las transiciones de características a clase
        for index, row in X_train.iterrows():
            for feature in X_train.columns:
                if row[feature] not in feature_counts[feature][y_train[index]]:
                    feature_counts[feature][y_train[index]][row[feature]] = 1
                else:
                    feature_counts[feature][y_train[index]][row[feature]] += 1

        # Calcular las probabilidades de transición de características a clase
        self.matriz_transicion = {}
        for feature in X_train.columns:
            self.matriz_transicion[feature] = {}
            for clase in self.clases:
                self.matriz_transicion[feature][clase] = {val: counts / sum(feature_counts[feature][clase].values()) for val, counts in feature_counts[feature][clase].items()}
                
                # self.matriz_transicion[feature][clase] = {val: counts / y_train.shape[0] for val, counts in feature_counts[feature][clase].items()}
            # print()
            # print(self.matriz_transicion) 

    def predict(self, X_test):
        predictions = []
        for index, row in X_test.iterrows():
            clase_probs = {clase: 1.0 for clase in self.clases}
            for feature in X_test.columns:
                for clase in self.clases:
                    if row[feature] in self.matriz_transicion[feature][clase]:
                        clase_probs[clase] *= self.matriz_transicion[feature][clase][row[feature]]
                    else:
                        clase_probs[clase] *= 1 / len(self.matriz_transicion[feature][clase])  # Laplace smoothing for unseen values
            
            predicted_class = max(clase_probs, key=clase_probs.get)
            predictions.append(predicted_class)
        return predictions

    def confusion_matrix(self, y_true, y_pred):
        y_true = list(y_true)
        matrix = {clase: {clase: 0 for clase in self.clases} for clase in self.clases}
        for i in range(len(y_true)):
            matrix[y_true[i]][y_pred[i]] += 1
        return matrix


df = pd.read_csv('monk-3-train.csv', index_col=False)
print(df)
print(df.columns)
target_column = df.columns[-1]

X, y = manage_df(df, target_column)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instanciar y ajustar el clasificador
clf = RedMarkovClasificacionCategorica()
clf.fit(X_train, y_train)

df_test = pd.read_csv('monk-3-test.csv', index_col=False)
target_column = df_test.columns[-1]

X_test, y_test = manage_df(df_test, target_column)

# Hacer predicciones
predictions = clf.predict(X_test)

# Calcular y mostrar la matriz de confusión
confusion_matrix = clf.confusion_matrix(y_test, predictions)
print("Matriz de Confusión:")
print(pd.DataFrame(confusion_matrix))

# Calcular y mostrar el accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Supongamos que quieres visualizar las transiciones de la primera característica a las clases
# feature_to_visualize = X_train.columns[3]  # Cambia esto según la característica que quieras visualizar

# print(clf.matriz_transicion)

