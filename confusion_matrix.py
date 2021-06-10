import numpy as np


class ImageConfusionMatrix:

    def __init__(self, y_true=[], y_pred=[], y_filenames=[]):

        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_filenames = y_filenames

        delta = np.subtract(y_true, y_pred)
        self.fn = np.count_nonzero(delta == 1)
        self.fp = np.count_nonzero(delta == -1)

        combo = np.add(y_true, y_pred)
        self.tp = np.count_nonzero(combo == 2)
        self.tn = np.count_nonzero(combo == 0)

        self.accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
        self.sensitivity = self.tp / (self.tp + self.fn)  # how good at detecting trues
        self.specificity = self.tn / (self.fp + self.tn)  # how good at detecting falses
        self.precision = self.tp / (self.tp + self.fp)   # type II error


if __name__ == '__main__':

    y_true = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    y_pred = [1, 0, 0, 0, 0, 1, 1, 1, 0, 1]
    labels = ["a", "b", "c", "d", "e", "f", "g"]

    c = ImageConfusionMatrix(y_true=y_true, y_pred=y_pred, y_filenames=labels)
    print(f"Accuracy: { c.accuracy }")
    print(f"Sensitivity: { c.sensitivity }")
    print(f"Specificity: {c.specificity}")
    print(f"Precision: {c.precision}")
