import cancer
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# get data
data = cancer.__load_data()
cancer_x = data.drop("target", axis=1).values
cancer_y = data['target'].values

for test_size in np.arange(0.1, 1, 0.1):
    error_rate = np.array([])
    for n in range(100):
        x_train, x_test, y_train, y_test = train_test_split(cancer_x, cancer_y,
                                                            test_size=test_size, shuffle=True)
        sc = StandardScaler()
        X_train_std = sc.fit_transform(x_train)
        X_test_std = sc.transform(x_test)

        lr = LogisticRegression()
        lr.fit(X_train_std, y_train)
        error_rate = np.append(error_rate, 1 - lr.score(X_test_std, y_test))
    print(f"Test Size > {test_size}")
    print(f"error mean: {np.mean(error_rate)}")
    print(f"error var: {np.var(error_rate)}")
