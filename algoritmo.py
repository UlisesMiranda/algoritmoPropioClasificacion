import pandas as pd
import numpy as np

def create_balanced_sample(dataset, target_column, sample_size=500):
    classes, counts = np.unique(dataset[target_column], return_counts=True)
    min_count = np.min(counts)
    
    balanced_sample = pd.DataFrame(columns=dataset.columns)
    np.random.seed(42)
    for cls in classes:
        cls_data = dataset[dataset[target_column] == cls]
        cls_indices = np.random.choice(cls_data.index, min_count, replace=False)
        balanced_sample = pd.concat([balanced_sample, dataset.loc[cls_indices]])
    
    indices = np.arange(len(balanced_sample))
    np.random.shuffle(indices)
    return balanced_sample.iloc[indices[:sample_size]], balanced_sample.iloc[indices[sample_size:]]

class RedMarkovClasificacionCategorica:
    def __init__(self, num_features):
        self.num_features = num_features
        self.selected_features = []
        self.matriz_transicion = None
        self.clases = None

    def calculate_mutual_information(self, X, y):
        mutual_info = []
        for column in X.columns:
            mutual_info.append((column, self.mutual_information(X[column], y)))
        return mutual_info

    def mutual_information(self, feature, target):
        feature_categories = np.unique(feature)
        target_categories = np.unique(target)
        mi = 0.0
        total_samples = len(target)

        for feat_cat in feature_categories:
            for tar_cat in target_categories:
                feat_target_count = np.sum((feature == feat_cat) & (target == tar_cat))
                feat_count = np.sum(feature == feat_cat)
                target_count = np.sum(target == tar_cat)

                joint_prob = feat_target_count / total_samples
                feat_prob = feat_count / total_samples
                target_prob = target_count / total_samples

                if joint_prob != 0 and feat_prob != 0 and target_prob != 0:
                    mi += joint_prob * np.log2(joint_prob / (feat_prob * target_prob))

        return mi

    def fit(self, X_train, y_train):
        self.clases = sorted(y_train.unique())

        mutual_info = self.calculate_mutual_information(X_train, y_train)
        mutual_info.sort(key=lambda x: x[1], reverse=True)

        self.selected_features = [info[0] for info in mutual_info[:self.num_features]]

        X_train_selected = X_train[self.selected_features]

        feature_counts = {}
        for feature in X_train_selected.columns:
            feature_counts[feature] = {clase: {} for clase in self.clases}

        for index, row in X_train_selected.iterrows():
            for feature in X_train_selected.columns:
                if row[feature] not in feature_counts[feature][y_train[index]]:
                    feature_counts[feature][y_train[index]][row[feature]] = 1
                else:
                    feature_counts[feature][y_train[index]][row[feature]] += 1

        self.matriz_transicion = {}
        for feature in X_train_selected.columns:
            self.matriz_transicion[feature] = {}
            for clase in self.clases:
                self.matriz_transicion[feature][clase] = {val: counts / sum(feature_counts[feature][clase].values()) for val, counts in feature_counts[feature][clase].items()}

    def predict(self, X_test):
        predictions = []
        for index, row in X_test.iterrows():
            clase_probs = {clase: 1.0 for clase in self.clases}
            for feature in self.selected_features:
                for clase in self.clases:
                    if row[feature] in self.matriz_transicion[feature][clase]:
                        clase_probs[clase] *= self.matriz_transicion[feature][clase][row[feature]]
                    else:
                        clase_probs[clase] *= 1 / len(self.matriz_transicion[feature][clase])
            
            predicted_class = max(clase_probs, key=clase_probs.get)
            predictions.append(predicted_class)
        return predictions

    def confusion_matrix(self, y_true, y_pred):
        y_true = list(y_true)
        matrix = {clase: {clase: 0 for clase in self.clases} for clase in self.clases}
        for i in range(len(y_true)):
            matrix[y_true[i]][y_pred[i]] += 1
        return matrix
    
    def confusion_matrix_to_array(self, conf_matrix):
        classes = sorted(conf_matrix.keys())
        num_classes = len(classes)

        conf_array = np.zeros((num_classes, num_classes), dtype=int)

        for i, true_class in enumerate(classes):
            for j, pred_class in enumerate(classes):
                conf_array[i, j] = conf_matrix[true_class][pred_class]

        return conf_array

    
    def calculate_metrics(self, y_true, y_pred, name_set):
        conf_matrix = self.confusion_matrix(y_true, y_pred)
        conf_array = self.confusion_matrix_to_array(conf_matrix)

        print(f"\nMatriz de Confusión - {name_set}:")
        print(pd.DataFrame(conf_matrix))

        true_positives = np.diag(conf_array)
        false_positives = np.sum(conf_array, axis=0) - true_positives
        false_negatives = np.sum(conf_array, axis=1) - true_positives

        precision = np.sum(true_positives / (true_positives + false_positives)) / len(self.clases)
        recall = np.sum(true_positives / (true_positives + false_negatives)) / len(self.clases)
        
        f1_score = 2 * (precision * recall) / (precision + recall)

        accuracy = true_positives.sum() / conf_array.sum()
        
        print(f"Accuracy - {name_set}: {accuracy}")
        print(f"Precision - {name_set}: {precision}")
        print(f"Recall - {name_set}: {recall}")
        print(f"F1 Score - {name_set}: {f1_score}")

        error = 1 - (sum(y_pred == y_true) / len(y_true))

        print(f"Error - {name_set}: {error}")

# Carga del conjunto de datos original
df = pd.read_csv('mushroom.csv', index_col=False)
target_column = df.columns[-1]

# Creación de la muestra aleatoria y balanceada
balanced_train_set, balanced_test_set = create_balanced_sample(df, target_column, sample_size=500)

# Conjunto de entrenamiento (500 ejemplos)
X_train = balanced_train_set.iloc[:, :-1]
y_train = balanced_train_set.iloc[:, -1]

# Conjunto de prueba (restantes no utilizados en el entrenamiento)
X_test = balanced_test_set.iloc[:, :-1]
y_test = balanced_test_set.iloc[:, -1]

# Número de características seleccionadas
num_top_features = 4

# Instanciar el modelo y ajustarlo
clf = RedMarkovClasificacionCategorica(num_features=num_top_features)
clf.fit(X_train, y_train)

# Nuestro modelo
print("\nMODELO:\n\n", clf.matriz_transicion)

# Predicciones para conjunto de entrenamiento y de prueba
train_predictions = clf.predict(X_train)
test_predictions = clf.predict(X_test)

# Cálculo de las métricas
clf.calculate_metrics(y_train, train_predictions, 'train')
clf.calculate_metrics(y_test, test_predictions, 'test')