import numpy as np

def svmkernel(x, kernel='gaussian', kerneloption=[1], xsup=None, frame=None, vector=None, dual=None):
    if xsup is None:
        xsup = x
    
    n1, n2 = x.shape
    n, n3 = xsup.shape
    
    ps = np.zeros((n1, n))
    
    if kernel.lower() == 'poly':
        nk, nk2 = kerneloption.shape if isinstance(kerneloption, np.ndarray) else (len(kerneloption), 1)
        
        if nk > nk2:
            kerneloption = kerneloption.T
            nk2 = nk
        
        if nk2 == 1:
            degree = kerneloption
            var = np.ones(n2)
        elif nk2 == 2:
            degree = kerneloption[0]
            var = np.ones(n2) * kerneloption[1]
        elif nk2 == n2 + 1:
            degree = kerneloption[0]
            var = kerneloption[1:]
        elif nk2 == n2 + 2:
            degree = kerneloption[0]
            var = kerneloption[1:n2+1]
        
        if nk2 == 1:
            aux = 1
        else:
            aux = np.tile(var, (n, 1))
        
        ps = np.dot(x, (xsup * aux**2).T)
        
        K = (ps + 1) ** degree if degree > 1 else ps
    
    elif kernel.lower() == 'polyhomog':
        nk, nk2 = kerneloption.shape if isinstance(kerneloption, np.ndarray) else (len(kerneloption), 1)
        
        if nk > nk2:
            kerneloption = kerneloption.T
            nk2 = nk
        
        if nk2 == 1:
            degree = kerneloption
            var = np.ones(n2)
        else:
            if nk2 != n2 + 1:
                degree = kerneloption[0]
                var = np.ones(n2) * kerneloption[1]
            else:
                degree = kerneloption[0]
                var = kerneloption[1:nk2]
        
        aux = np.tile(var, (n, 1))
        ps = np.dot(x, (xsup * aux**2).T)
        
        K = ps ** degree

    
    elif kernel.lower() == 'gaussian':
        nk, nk2 = kerneloption.shape if isinstance(kerneloption, np.ndarray) else (len(kerneloption), 2)
        
        if nk != nk2:
            if nk > nk2:
                kerneloption = kerneloption.T
                nk2 = nk
            else:
                kerneloption = np.ones(n2) * kerneloption[0]
                nk2 = n2
        
        if nk2 != n2 and nk2 != n2 + 1:
            raise ValueError('Number of kerneloption is not compatible with data...')
        elif nk2 == n2:
            metric = np.diag(1/kerneloption**2)
        else:
            metric = np.diag(1/kerneloption[1:]**2)
        
        ps = np.dot(x, metric).dot(xsup.T)
        normx = np.sum(x**2 * metric, axis=1)
        normxsup = np.sum(xsup**2 * metric, axis=1)
        ps = -2*ps + normx[:, np.newaxis] + normxsup[np.newaxis, :]
        K = np.exp(-ps/2)
    
    elif kernel.lower() == 'htrbf':
        a, b = kerneloption[:2]
        ps = np.zeros((n1, n))
        
        for i in range(n):
            ps[:, i] = np.sum(np.abs(x**a - xsup[i, np.newaxis]**a)**b, axis=1)
        
        K = np.exp(-ps)
    
    elif kernel == 'gaussianslow':
        for i in range(n):
            ps[:,i] = np.sum(np.abs((x - np.ones((n1,1))*xsup[i,:]))**2, axis=1)/(kerneloption**2*2)
        K = np.exp(-ps)
        
    elif kernel == 'multiquadric':
        metric = np.diag(1./kerneloption)
        ps = x @ metric @ xsup.T
        normx = np.sum(x**2 * metric, axis=1)
        normxsup = np.sum(xsup**2 * metric, axis=1)
        ps = -2*ps + normx.reshape(-1,1) + normxsup
        K = np.sqrt(ps + 0.1)
        
    elif kernel == 'wavelet':
        K = kernelwavelet(x, kerneloption, xsup)
        
    elif kernel == 'frame':
        K = kernelframe(x, kerneloption, xsup, framematrix, vector, dual)
        
    elif kernel == 'wavelet2d':
        K = wav2dkernelint(x, xsup, kerneloption)
        
    elif kernel == 'radialwavelet2d':
        K = radialwavkernel(x, xsup)
        
    elif kernel == 'tensorwavkernel':
        K, option = tensorwavkernel(x, xsup, kerneloption)
        
    elif kernel == 'numerical':
        K = kerneloption['matrix']
        
    elif kernel == 'polymetric':
        K = x @ kerneloption['metric'] @ xsup.T
        
    elif kernel == 'jcb':
        K = x @ xsup.T
        
    else:
        raise ValueError('Invalid kernel type specified.')
    
    return K, kerneloption

    


