import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , r2_score


TF = "./Data/Transformed.csv"
TRAINF = "./Data/History.xlsx"
TESTF =  "./Data/Fcst.xlsx"
MF = "./Model.pkl"

def ReadData(FP):
    df = pd.read_excel(FP)
    df["month"] = df["DATE"].dt.month
    df["year"] = df["DATE"].dt.year
    df["prevmonth"] = df["DATE"].dt.month - 1
    df["prevyear"] = df["DATE"].dt.year - 1
    df["rollingLoad5"] = df["Load"].rolling(5).mean()
    df["rollingLoad10"] = df["Load"].rolling(10).mean()

    df['prevMonthLoad'] = df.groupby('prevmonth')['Load'].transform(np.mean)
    df['prevYearLoad'] = df.groupby('prevyear')['Load'].transform(np.mean)
    df = df.fillna(0)
    print(df.head())
    df.to_csv(TF)

def Model():
    scaler = StandardScaler()
    df = pd.read_csv(TF)
    df = df[["Hour" ,  "Temperature" ,"Load"]]
    df = scaler.fit_transform(df)

    X = df[:,0:2]
    y = df[:,2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    model = LinearRegression()
    r = model.fit(X_train,y_train)
    preds = model.predict(X_test)
    score = r2_score(y_test , preds)
    print(score)
    with open(MF , "wb") as fp:
        pickle.dump(r , fp)

def Forecast():
    testdf = pd.read_excel(TESTF)
    traindf = pd.read_excel(TRAINF)
    df = pd.concat([traindf , testdf])
    #testdf[0,:]["rollingLoad5"] =
    for r in testdf.iterrows():
        print(r)
        r["rollingLoad5"] = df.rolling(5).mean()
        r["rollingLoad10"] = df["Load"].rolling(10).mean()

        r['prevMonthLoad'] = df.groupby('prevmonth')['Load'].transform(np.mean)
        r['prevYearLoad'] = df.groupby('prevyear')['Load'].transform(np.mean)
    #with open(MF , "rb") as fp:
    #    model = pickle.load(fp)


def Run():
    #ReadData(TRAINF)
    Model()
    #Forecast()



if __name__ == "__main__":

    Run()
