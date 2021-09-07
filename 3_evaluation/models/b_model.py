import pandas as pd
import numpy as np
import random
import random
np.random.seed(1234)
random.seed(1234)

from sklearn import linear_model


class b_Model():

    model = None

    def fit(self, x, y):
        model = linear_model.LinearRegression()
        model.fit(x, y)
        self.model = model

    def predict(self, x):
        return self.model.predict(x)


b_model = b_Model()
b_data = pd.read_csv('data/b_train.csv')
b_features = b_data[b_data.columns.difference(['flight_length_minutes'])]
b_target = b_data['flight_length_minutes'].astype(int).values
b_data.head()
b_model.fit(b_features, b_target)

