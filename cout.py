import numpy as np

def cout(H, x, y, C, ind, c, A, b, lambd):
    n, m = H.shape
    X = np.zeros((n, 1))
    posok = np.where(ind > 0)[0]
    posA = np.where(ind == 0)[0]     # liste des contriantes saturees  
    posB = np.where(ind == -1)[0]    # liste des contriantes saturees  
    X[posok] = x  
    X[posB] = C[posB]
    
    J = 0.5 * np.dot(np.dot(X.T, H), X) - np.dot(c.T, X)
    
    lam = 0
    
    return J, lam
