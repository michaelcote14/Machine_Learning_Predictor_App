
def imports():
    import pandas as pd
    import sklearn
    from sklearn import linear_model
    import numpy as np
    from sklearn import metrics
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
from Variables import *




def GetCoefficients():
    print("(Feature)",
          "[Coefficient Value] *The Coefficient below is the correlators of your current data picks, while the corr method above is the correlators of the entire data set")
    for index, feature in enumerate(DataPicks):
        try:
            print("(", feature, ")", "[", MyLinearRegression.coef_[index], "]")
        except:
            pass
GetCoefficients()

