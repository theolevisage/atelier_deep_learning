from sklearn import datasets, model_selection, linear_model, tree
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

textual_numbers = datasets.load_digits().target_names
print(textual_numbers)

digits = datasets.load_digits()
flat_4 = digits.images[4].flatten()
print(flat_4)
print(digits.images[4])


plt.figure(0, figsize=(3, 3))
plt.imshow(digits.images[0], cmap=plt.cm.gray_r)
plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[1], cmap=plt.cm.gray_r)
print(type(digits.images))
index = 2
for digit in digits.images[:12, :]:
    plt.figure(index, figsize=(3, 3))
    plt.imshow(digit, cmap=plt.cm.gray_r)
    index += 1

plt.show()

X, y = datasets.load_digits(return_X_y=True, as_frame=True)

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.4)

RFclf = RandomForestClassifier()
RFclf.fit(x_train, y_train)

y_pred = RFclf.predict(x_test)

result = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result)
precision = RFclf.score(x_test, y_test)
print("Précision : ", precision * 100)

rn = random.randrange(0, 1001)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('PyCharm')
