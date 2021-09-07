import numpy as np

class a_Model():

    def __init__(self):
        return None

    def _sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def predict(self, x):
        col_1 = x[:, 0]
        col_2 = x[:, 1]
        col_3 = x[:, 2]
        col_4 = x[:, 3]
        col_5 = x[:, 4]
        col_6 = x[:, 5]
        col_7 = x[:, 6]
        col_8 = x[:, 7]

        res = np.sqrt(np.abs(((-1) * (col_2 / ((0.1) * col_3 / 100)) + 13 * col_4 + np.sqrt(100 * col_8) - col_7) * col_8))
        pred_prob_true = self._sigmoid(res - 20)

        pred_prob_false = 1-pred_prob_true
        pred_prob_false = pred_prob_false[np.newaxis].transpose()
        pred_prob_true = pred_prob_true[np.newaxis].transpose()
        # write them next to each other
        res = np.append(pred_prob_false, pred_prob_true, axis=1)
        return res[:, 1] > 0.5

a_model = a_Model()

