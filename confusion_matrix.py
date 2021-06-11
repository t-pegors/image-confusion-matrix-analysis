import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import random

class ImageConfusionMatrix:

    def __init__(self, image_path, y_true=[], y_pred=[], y_filenames=[]):

        self.df = pd.DataFrame()
        self.path = image_path
        self.df['filenames'] = y_filenames
        self.df['y_true'] = y_true
        self.df['y_pred'] = y_pred

        # build confusion matrix
        col1, col2 = 'y_true', 'y_pred'
        conditions = [(self.df[col1] == 1) & (self.df[col2] == 1),
                      (self.df[col1] == 0) & (self.df[col2] == 0),
                      (self.df[col1] == 1) & (self.df[col2] == 0),
                      (self.df[col1] == 0) & (self.df[col2] == 1)]
        choices = ['tp', 'tn', 'fn', 'fp']
        self.df['conf_mat'] = np.select(conditions, choices, default=np.nan)
        self.cm = json.loads(self.df['conf_mat'].value_counts().to_json())

        # calculate measures
        self.accuracy = round((self.cm['tp'] + self.cm['tn']) / (self.cm['tp'] + self.cm['tn'] + self.cm['fp'] + self.cm['fn']), 2)
        self.sensitivity = round(self.cm['tp'] / (self.cm['tp'] + self.cm['fn']), 2)
        self.specificity = round(self.cm['tn'] / (self.cm['fp'] + self.cm['tn']), 2)
        self.precision = round(self.cm['tp'] / (self.cm['tp'] + self.cm['fp']), 2)

    def matplotlib_cm_display(self, type_list=[]):

        titles = {'tp': 'True Positives',
                  'tn': 'True Negatives',
                  'fp': 'False Positives',
                  'fn': 'False Negatives'}
        if type_list[0] == 'all':
            type_list = ['tp', 'tn', 'fp', 'fn']

        for type in type_list:

            file_set = list(self.df.loc[self.df['conf_mat'] == type]['filenames'])

            if len(file_set) == 0:
                print("No images found in this category.")
                return
            if len(file_set) > 4:
                file_set = random.sample(file_set, 4)

            print(file_set)

            plt.figure(num=titles[type])
            for file_num in range(0,len(file_set)):
                I = cv2.imread(os.path.join(self.path, file_set[file_num]))
                I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

                if I.shape[0] > 200:
                    percent_redux = round(200 / I.shape[0], 2)
                    width = int(I.shape[1] * percent_redux)
                    height = int(I.shape[0] * percent_redux)
                    I = cv2.resize(I, (width, height))

                title = self.df.loc[file_num, 'filenames']
                # Row, column, index
                plt.subplot(2, 2, file_num + 1)
                plt.imshow(I)
                plt.title(title, fontsize=8)
                plt.xticks([])
                plt.yticks([])

        plt.show()


if __name__ == '__main__':

    path = 'YOUR_IMAGES_PATH_HERE'
    valid_images = [".jpg", ".jpeg", ".gif", ".png"]

    filenames = []
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        filenames.append(f)

    #y_true = list(np.random.randint(2, size=len(filenames)))
    #y_pred = list(np.random.randint(2, size=len(filenames)))
    y_true = [1,1,1,1,0,0,0,0]
    y_pred = [1,1,0,0,1,1,0,0]

    c = ImageConfusionMatrix(path, y_true=y_true, y_pred=y_pred, y_filenames=filenames)
    print(f"Accuracy: { c.accuracy }")
    print(f"Sensitivity: { c.sensitivity }")
    print(f"Specificity: {c.specificity}")
    print(f"Precision: {c.precision}")

    c.matplotlib_cm_display(type_list=['all'])
