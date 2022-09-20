<<<<<<< HEAD

def imports():
    import pandas as pd
    import sklearn
    from sklearn import linear_model
    import numpy as np
    from sklearn import metrics
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
=======
>>>>>>> 81cb0a029fb1ba650c5e57bffd11a22dcaea02f5
from Variables import *




def GetCoefficients():
    print("(Feature)",
          "[Coefficient Value] *The Coefficient below is the correlators of your current data picks, while the corr method above is the correlators of the entire data set")
    for index, feature in enumerate(DataPicks):
        try:
            print("(", feature, ")", "[", MyLinearRegression.coef_[index], "]")
        except:
            pass
<<<<<<< HEAD
GetCoefficients()

=======
GetCoefficients()
>>>>>>> 81cb0a029fb1ba650c5e57bffd11a22dcaea02f5
