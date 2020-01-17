import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def ReadData():
    df = pd.read_excel("./Data/History.xlsx")



def Run():
    ReadData()



if __name__ == "__main__":

    Run()
