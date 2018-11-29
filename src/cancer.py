import pandas as pd
from sklearn.datasets import load_breast_cancer

def __load_data():
    """
    :return: tuple(train_data, test_data)
    """
    cancer = load_breast_cancer()
    df_origin = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df_origin['target'] = cancer.target

    return df_origin
