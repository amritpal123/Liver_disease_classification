# for numerical computing
import numpy as np

# for dataframes
import pandas as pd
import joblib
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
import os
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.2.0-posix-seh-rt_v5-rev0\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']


def partition1(x):
    if x == " no_disease":
        return 0
    elif x == " suspect_disease":
        return 1
    elif x == " hepatitis":
        return 2
    elif x == " fibrosis":
        return 3
    elif x==' cirrhosis':
        return 4

def remap(output):
    if output == 0:
        return "Person is healthy"
    elif output == 1:
        return  "Person has a liver disease but it's not hepatitis, cirrhosis and fibrosis."
    elif output == 2:
        return "hepatitis"
    elif output == 3:
        return "fibrosis"
    else:
        return " cirrhosis"

def training():
    df=pd.read_csv("C:\\Users\\Amritpal Singh\\Downloads\\HCV-data.csv")

    df['Dataset'] = df['Dataset'].map(partition1)


    df = df[df.aspartate_aminotransferase <=2500]

    df=df.dropna(axis=0, how='any')

    # Create separate object for target variable
    y = df.Dataset

    # Create separate object for input features
    X = df[['creatinine', 'bilirubin', 'aspartate_aminotransferase', 'gamma_glutamyl_transferase ']]


    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=1234)

    clf = GaussianNB()
    clf.fit(X_train, y_train)

    joblib.dump(clf, 'D:/Minor-20211113T152548Z-001/Minor/GNB.sav')

training()


def predict_out_GNB(test):

    print(test)

    model = joblib.load('D:/Minor-20211113T152548Z-001/Minor/GNB.sav')

    X_pred = list(test.values())

    for i in range(len(X_pred)):
        X_pred[i]=float(X_pred[i])

    out = model.predict([X_pred])

    out = remap(out)
    return out