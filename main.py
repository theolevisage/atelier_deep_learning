from sklearn import datasets, model_selection, linear_model, tree
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt

textual_numbers = datasets.load_digits().target_names
print(textual_numbers)

digits = datasets.load_digits()
flat_4 = digits.images[4].flatten()
print(flat_4)

plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[4], cmap=plt.cm.gray_r, interpolation="bessel")

plt.show()

X, y = datasets.load_digits(return_X_y=True, as_frame=True)

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.4)

rn = random.randrange(0, 1001)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('PyCharm')
