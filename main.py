import numpy as np
from sklearn import datasets, model_selection
import random
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

textual_numbers = datasets.load_digits().target_names
print(textual_numbers)

digits = datasets.load_digits()

flat_list = []
for digit in digits.images:
    flat_list.append(digit.flatten())

plt.figure(0, figsize=(3, 3))
plt.imshow(digits.images[0])
plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[1])
index = 2

for digit in digits.images[:4, :]:
    plt.figure(index, figsize=(3, 3))
    plt.imshow(digit)
    index += 1

X, y = datasets.load_digits(return_X_y=True, as_frame=True)

x_train, x_test, y_train, y_test = model_selection.train_test_split(X.values, y, test_size=0.4)

RFclf = RandomForestClassifier()
RFclf.fit(x_train, y_train)

y_pred = RFclf.predict(x_test)

clf = MLPClassifier(random_state=1).fit(x_train, y_train)

new_number = np.array([[0, 0, 5, 13, 9, 1, 0, 0],
              [0, 0, 13, 15, 10, 15, 5, 0],
              [0, 3, 15, 2, 0, 9, 11, 0],
              [0, 4, 11, 0, 0, 6, 10, 0],
              [0, 0, 0, 12, 10, 9, 8, 0],
              [0, 0, 0, 0, 0, 12, 10, 0],
              [0, 0, 0, 1, 4, 12, 6, 0],
              [0, 0, 0, 11, 10, 5, 0, 0]]).flatten()

print("Test : ")
print(x_test[:1])
#nn_prediction = clf.predict(x_test[:1])
nn_prediction = clf.predict([new_number])
print(nn_prediction)

result = classification_report(y_test, y_pred)
print("Classification Report:", )
print(result)
precision = RFclf.score(x_test, y_test)
print("Précision : ", precision * 100)

rn = random.randrange(0, 1001)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('PyCharm')

plt.show()
