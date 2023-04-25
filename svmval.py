import numpy as np
from svmkernel import svmkernel
from fileaccess import fileaccess

def svmval(x, xsup, w, b, kernel='gaussian', kerneloption=1, span=None, framematrix=None, vector=None, dual=None):

    semiparam = 0
    
    if xsup.__class__.__name__ != 'dict':
        nsup, nd = xsup.shape
    else:
        nsup = len(xsup['indice'])
        nd = xsup['dimension']
    
    if x.__class__.__name__ != 'dict':
        nl, nc = x.shape
    else:
        nl = len(x['indice'])
        nc = x['dimension']
    
    if nc != nd:
        raise ValueError('x and xsup must have the same number of columns')
    
    if span is not None:
        semiparam = 1
    
    if kernel != 'frame' or framematrix is None or vector is None:
        framematrix = None
        vector = None
    
    if dual is None:
        dual = None
        
    if kernel != 'numerical' and x.__class__.__name__ != 'dict' and xsup.__class__.__name__ != 'dict' and (nl > 1000 or nsup > 1000):
        if w.size != 0:
            chunksize = 100
            chunks1 = int(np.ceil(nsup/chunksize))
            chunks2 = int(np.ceil(nl/chunksize))
            y2 = np.zeros((nl,1))
            for ch1 in range(chunks1):
                ind1 = slice(1+(ch1-1)*chunksize, min(nsup, ch1*chunksize))
                for ch2 in range(chunks2):
                    ind2 = slice(1+(ch2-1)*chunksize, min(nl, ch2*chunksize))
                    kchunk = svmkernel(x[ind2,:], kernel, kerneloption, xsup[ind1,:])
                    y2[ind2] += np.dot(kchunk, w[ind1])
            if semiparam:
                y1 = span * b
                y = y1 + y2
            else:
                y = y2 + b
        else:
            y = np.array([])
    elif 'datafile' in xsup or 'datafile' in x:
        if xsup.__class__.__name__ == 'dict':
            nsup = len(xsup['indice'])
        else:
            nsup = xsup.shape[0]
        if x.__class__.__name__ == 'dict':
            nl = len(x['indice'])
        else:
            nl = x.shape[0]
        chunksize = 100
        chunks1 = int(np.ceil(nsup/chunksize))
        chunks2 = int(np.ceil(nl/chunksize))
        y2 = np.zeros((nl,1))
        for ch1 in range(chunks1):
            ind1 = slice(1+(ch1-1)*chunksize, min(nsup, ch1*chunksize))
            for ch2 in range(chunks2):
                ind2 = slice(1+(ch2-1)*chunksize, min(nl, ch2*chunksize))
                if 'datafile' not in x:
                    x1 = x[ind2,:]
                else:
                    x1 = fileaccess(x['datafile'], x['indice'][ind2], x['dimension'])
                if 'datafile' not in xsup:
                    x2 = xsup[ind1,:]
                else:
                    x2 = fileaccess(xsup['datafile'], xsup['indice'][ind1], xsup['dimension'])
            
                    kchunk = svmkernel(x1, kernel, kerneloption, x2)
                        # kchunk = svmkernel(x[ind2, :], kernel, kerneloption, xsup[ind1, :])
            
                    y2[ind2] += kchunk @ w[ind1]
    else:
        ps = svmkernel(x, kernel, kerneloption, xsup, framematrix, vector, dual)
        if semiparam:
            y1 = span * b
            if w.size == 0:
                y = y1
                y2 = np.zeros_like(y1)
            else:
                y2 = np.dot(ps, w)
                y = y1 + y2
        else:
            y = np.dot(ps, w) + b

    return y, y1, y2
