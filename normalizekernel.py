import numpy as np
from svmkernel import svmkernel

def normalizekernel(x, kernel, kerneloption, xsup=None):
    n = x.shape[0]
    
    kdiag1 = np.diag(svmkernel(x, kernel, kerneloption))
    
    if xsup is not None:
        kdiag2 = np.diag(svmkernel(xsup, kernel, kerneloption))
        Kp = svmkernel(x, kernel, kerneloption, xsup)
    else:
        kdiag2 = kdiag1
        Kp = svmkernel(x, kernel, kerneloption)
    
    Kweight = kdiag1.reshape(-1, 1) * kdiag2.reshape(1, -1)
    K = Kp / np.sqrt(Kweight)
    
    return K
