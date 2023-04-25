import numpy as np
from svmutil import *

def svmoneclass(x, kernel, kerneloption, nu, verbose, alphainit=None):
    n1, n2 = x.shape if len(x) > 0 else (0, 0)
    K = normalizekernel(x, kernel, kerneloption)
    c = np.zeros(n1)
    A = np.ones(n1)
    b = 1
    C = 1 / nu / n1
    lambd = 1e-8
    if alphainit is None:
        alphainit = C / 2 * np.ones(n1)
    else:
        alphainit[alphainit >= C] = C
    alpha, multiplier, pos = svm_train([], alphainit, K, "-s 2 -t {} -q".format(kernel))
    xsup = x[pos, :] if len(x) > 0 else []
    Ksup = K[pos][:, pos]
    return xsup, alpha, multiplier, pos, Ksup
