import cancer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# get data
data = cancer.__load_data()
cancer_x = data.drop("target", axis=1).values
cancer_y = data['target'].values

error_mean = np.array([])
error_var = np.array([])
for split_num in range(2, 10, 1):
    error_rate = np.array([])
    kfold = KFold(n_splits=split_num, shuffle=True)
    for n in range(100):
        result = cross_val_score(LogisticRegression(), cancer_x, cancer_y,
                                 cv=kfold, scoring="accuracy")

        error_rate = np.append(error_rate, 1 - np.mean(result))

    error_mean = np.append(error_mean, np.mean(error_rate))
    error_var = np.append(error_var, np.var(error_rate))
    print(f"split num > {split_num}")
    print(f"error mean: {np.mean(error_rate)}")
    print(f"error var: {np.var(error_rate)}")

plt.plot(range(2, 10, 1), error_mean, color='k', linestyle='--')
plt.plot(range(2, 10, 1), error_var, color='g')
plt.show()
