from sklearn import datasets, model_selection, linear_model, tree
from sklearn.linear_model import LogisticRegression
import pandas as pd
import random

type_iris = datasets.load_iris().target_names
print(type_iris)

X, y = datasets.load_iris(return_X_y=True, as_frame=True)

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.4)

rn = random.randrange(0, 1001)

modele_reg = linear_model.LogisticRegression(max_iter=1000, random_state=rn)
modele_reg.fit(x_train.values, y_train)

modele_tree = tree.DecisionTreeClassifier(random_state=rn)
modele_tree.fit(x_train.values, y_train)

precision_reg = modele_reg.score(x_test.values, y_test)
print("Précision du modèle de régression : ", precision_reg * 100)

precision_tree = modele_tree.score(x_test.values, y_test)
print("Précision du modèle d'arbre de classification : ", precision_tree * 100)

# Prédiction d'une iris
# modele_to_predict = [[5.1, 3.5, 1.4, 0.2]]  # setosa from dataset
# modele_to_predict = [[7.0, 3.2, 4.7, 1.4]]  # versicolor from dataset
# modele_to_predict = [[6.3, 2.9, 5.6, 1.8]]  # virginica from dataset
# modele_to_predict = [[7.0, 2.5, 4.2, 1.2]]  # random from us
# modele_to_predict = [[5.4, 2.8, 3.2, 1.6]]  # random from us
modele_to_predict = [[6.1, 2.4, 5.9, 2.1]]  # random from us
prediction = modele_reg.predict(modele_to_predict)
print("Prédiction reg obtenu : " + type_iris[prediction[0]])
prediction = modele_tree.predict(modele_to_predict)
print("Prédiction tree obtenu : " + type_iris[prediction[0]])

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('PyCharm')
