import numpy as np

def DataGrid2D(vectX, vectY):
    xtest1, xtest2 = np.meshgrid(vectX, vectY)
    nn1, nn2 = xtest1.shape
    xtest = np.hstack((xtest1.reshape(nn1 * nn2, 1), xtest2.reshape(nn1 * nn2, 1)))
    return xtest, xtest1, xtest2, nn1, nn2
