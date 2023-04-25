import numpy as np
from normalizekernel import normalizekernel

def svmoneclassval(x, xsup, alpha, rho, kernel, kerneloption):
    K = normalizekernel(x, kernel, kerneloption, xsup)
    ypred = np.dot(K, alpha) + rho
    return ypred