import numpy as np
def carpette(x):
    color = np.ones(len(x))
    r2 = x[:,0]**2 + x[:,1]**2
    r2[r2<0.04] = 10
    c2 = np.where(r2<0.25)[0]
    color[c2[np.where(x[c2,0]>=0)[0]]] = -1
    r2[c2] = 10
    c2 = np.where(r2<0.64)[0]
    color[c2[np.where(x[c2,0]<0)[0]]] = -1
    r2[c2] = 10
    c2 = np.where(r2<10)[0]
    #color[c2[np.where(x[c2,1]<0)[0]]] = 3
    #color[c2[np.where(x[c2,1]>=0)[0]]] = 4
    return color

def datasets(n, nbapp, nbtest, sigma=0.4):
    xapp = np.array([])
    yapp = np.array([])
    xtest = np.array([])
    ytest = np.array([])
    xtest1 = np.array([])
    xtest2 = np.array([])

    if n.lower() == 'mixture':
        # Data Learn Generation
        nbapp = round(nbapp/3)
        x1 = sigma*np.random.randn(nbapp) + 0.3
        x2 = sigma*np.random.randn(nbapp) - 0.3
        x3 = sigma*np.random.randn(nbapp) - 1
        y1 = sigma*np.random.randn(nbapp) + 0.5
        y2 = sigma*np.random.randn(nbapp) - 0.5
        y3 = sigma*np.random.randn(nbapp) - 1

        xapp = np.vstack((np.hstack((x1, x2)), x3)).T
        yapp = np.hstack((np.ones(nbapp), -1*np.ones(nbapp), np.ones(nbapp)))

        # Data Test Generation
        nbtest = round(nbtest/3)
        xt = sigma*np.random.randn(nbtest) + 0.3
        xt2 = sigma*np.random.randn(nbtest) - 0.3
        xt3 = sigma*np.random.randn(nbtest) - 1
        yt = sigma*np.random.randn(nbtest) + 0.5 + 0.7*xt
        yt2 = sigma*np.random.randn(nbtest) - 0.5
        yt3 = sigma*np.random.randn(nbtest) - 1 - 0.7*xt
        xtest = np.vstack((np.hstack((xt, xt2)), xt3)).T
        ytest = np.hstack((np.ones(nbtest), -1*np.ones(nbtest), np.ones(nbtest)))

    elif n.lower() == 'gaussian':
        nbapp = round(nbapp/2)
        nbtest = round(nbtest/2)
        if sigma is None:
            sigma = 0.2
        x1 = sigma * np.random.randn(1, nbapp) + 0.3
        x2 = sigma * np.random.randn(1, nbapp) - 0.3
        y1 = sigma * np.random.randn(1, nbapp) + 0.5
        y2 = sigma * np.random.randn(1, nbapp) - 0.5
        xapp = np.vstack((np.hstack((x1, x2)), np.hstack((y1, y2)))).T
        yapp = np.hstack((np.ones(nbapp), -np.ones(nbapp)))
        x1 = sigma * np.random.randn(1, nbtest) + 0.3
        x2 = sigma * np.random.randn(1, nbtest) - 0.3
        y1 = sigma * np.random.randn(1, nbtest) + 0.5
        y2 = sigma * np.random.randn(1, nbtest) - 0.5
        xtest = np.vstack((np.hstack((x1, x2)), np.hstack((y1, y2)))).T
        ytest = np.hstack((np.ones(nbtest), -np.ones(nbtest)))

    elif n.lower() == 'checkers':
        xapp = []
        yapp = []
        nb = np.floor(nbapp/16).astype(int)
        for i in range(-2, 2+1):
            for j in range(-2, 2+1):
                xapp = np.vstack((xapp, np.hstack((i+np.random.rand(nb,1), j+np.random.rand(nb,1)))))
                yapp = np.hstack((yapp, (2*np.remainder((i+j+4),2)-1)*np.ones(nb)))
        xtest = []
        ytest = []
        nb = np.floor(nbtest/16).astype(int)
        for i in range(-2, 2+1):
            for j in range(-2, 2+1):
                xtest = np.vstack((xtest, np.hstack((i+np.random.rand(nb,1), j+np.random.rand(nb,1)))))
                ytest = np.hstack((ytest, (2*np.remainder((i+j+4),2)-1)*np.ones(nb)))

    elif n.lower() == 'clowns':
        nbapp = round(nbapp / 2)
        nbtest = round(nbtest / 2)
        
        x1 = (6 * np.random.rand(nbapp, 1) - 3)
        x2 = x1 ** 2 + np.random.randn(nbapp, 1)
        x0 = sigma * np.random.randn(nbapp, 2) + (np.ones((nbapp, 1)) * [0, 6])
        xapp = np.concatenate((x0, np.concatenate((x1, x2), axis=1)), axis=0)
        xapp = (xapp - np.ones((2 * nbapp, 1)) * np.mean(xapp, axis=0)) * np.diag(1. / np.std(xapp, axis=0))
        yapp = np.concatenate((np.ones((nbapp, 1)), -np.ones((nbapp, 1))), axis=0)
        
        if nbtest > 0:
            x1 = (6 * np.random.rand(nbtest, 1) - 3)
            x2 = x1 ** 2 + np.random.randn(nbtest, 1)
            x0 = sigma * np.random.randn(nbtest, 2) + (np.ones((nbtest, 1)) * [0, 6])
            xtest = np.concatenate((x0, np.concatenate((x1, x2), axis=1)), axis=0)
            xtest = (xtest - np.ones((2 * nbtest, 1)) * np.mean(xtest, axis=0)) * np.diag(1. / np.std(xtest, axis=0))
            ytest = np.concatenate((np.ones((nbtest, 1)), -np.ones((nbtest, 1))), axis=0)

    elif n.lower() =='cosexp':
        maxx = 5
        freq = 0.7
        nbtest = int(np.sqrt(nbtest))
        xi = np.random.rand(nbapp, 2)
        yapp = np.sign(np.cos(0.5 * (np.exp(freq * maxx * xi[:, 0]))) - (2 * xi[:, 1] - 1))
        xapp = np.concatenate((maxx * xi[:, 0:1], 2 * xi[:, 1:2] - 1), axis=1)
        xtest1, xtest2 = np.meshgrid(np.linspace(0, maxx, nbtest), np.linspace(-1, 1, nbtest))
        nn = len(xtest1)
        xtest = np.concatenate((xtest1.reshape(nn * nn, 1), xtest2.reshape(nn * nn, 1)), axis=1)
        ytest = np.sign(np.cos(0.5 * (np.exp(freq * maxx * xtest[:, 0]))) - (2 * xtest[:, 1] - 1))


    elif n.lower() == 'multiclasscheckers':
        xapp = []
        yapp = []
        nb = nbapp // 16
        for i in range(-2, 2+1):
            for j in range(-2, 2+1):
                xapp = np.concatenate((xapp, np.column_stack((i + np.random.rand(nb,1), j + np.random.rand(nb,1)))))
                if (abs(i) % 2 == 0) and (abs(j) % 2 == 0):
                    yapp = np.concatenate((yapp, np.ones((nb,1))))
                elif (abs(i) % 2 == 1) and (abs(j) % 2 == 0):
                    yapp = np.concatenate((yapp, 2*np.ones((nb,1))))
                elif (abs(i) % 2 == 0) and (abs(j) % 2 == 1):
                    yapp = np.concatenate((yapp, 3*np.ones((nb,1))))
                elif (abs(i) % 2 == 1) and (abs(j) % 2 == 1):
                    yapp = np.concatenate((yapp, 4*np.ones((nb,1))))
        xtest = []
        xt1, xt2 = np.meshgrid(np.linspace(-2, 2, int(np.sqrt(nbtest))))
        xt1 = xt1.reshape(-1, 1)
        xt2 = xt2.reshape(-1, 1)
        xtest = np.column_stack((xt1, xt2))
        ytest = np.zeros((len(xtest),1))
        for i in range(-2, 2+1):
            for j in range(-2, 2+1):
                pos = np.where((xtest[:,0] >= i) & (xtest[:,0] < i+1) & (xtest[:,1] >= j) & (xtest[:,1] < j+1))[0]
                if (abs(i) % 2 == 0) and (abs(j) % 2 == 0):
                    ytest[pos] = np.ones((len(pos),1))
                elif (abs(i) % 2 == 1) and (abs(j) % 2 == 0):
                    ytest[pos] = 2*np.ones((len(pos),1))
                elif (abs(i) % 2 == 0) and (abs(j) % 2 == 1):
                    ytest[pos] = 3*np.ones((len(pos),1))
                elif (abs(i) % 2 == 1) and (abs(j) % 2 == 1):
                    ytest[pos] = 4*np.ones((len(pos),1))
        xtest1 = xt1.reshape(xt1.shape[0], xt2.shape[0])
        xtest2 = xt2.reshape(xt1.shape[0], xt2.shape[0])
    
    elif n.lower()  == 'multiclassgaussian':
        mean1 = [1, 1]
        mean2 = [-1, 1]
        mean3 = [0, -1]
        x1 = sigma * np.random.randn(nbapp, 2) + np.ones((nbapp, 1)) * mean1
        y1 = np.ones((nbapp, 1))
        x2 = sigma * np.random.randn(nbapp, 2) + np.ones((nbapp, 1)) * mean2
        y2 = 2 * np.ones((nbapp, 1))
        x3 = sigma * np.random.randn(nbapp, 2) + np.ones((nbapp, 1)) * mean3
        y3 = 3 * np.ones((nbapp, 1))
        Ytarget = [1, 2, 3]
        xapp = np.vstack((x1, x2, x3))
        yapp = np.vstack((y1, y2, y3))
        nbapp = xapp.shape[0]
        x1 = sigma * np.random.randn(nbtest, 2) + np.ones((nbtest, 1)) * mean1
        y1 = np.ones((nbtest, 1))
        x2 = sigma * np.random.randn(nbtest, 2) + np.ones((nbtest, 1)) * mean2
        y2 = 2 * np.ones((nbtest, 1))
        x3 = sigma * np.random.randn(nbtest, 2) + np.ones((nbtest, 1)) * mean3
        y3 = 3 * np.ones((nbtest, 1))
        Ytarget = [1, 2, 3]
        xtest = np.vstack((x1, x2, x3))
        ytest = np.vstack((y1, y2, y3))
        nbapp = xapp.shape[0]

    elif n.lower()  == 'westonnonlinear':
        nbapp = round(nbapp / 2)
        yapp = np.vstack((np.ones((nbapp, 1)), -1 * np.ones((nbapp, 1))))
        A = 2 * (np.random.rand(2 * nbapp, 1) < 0.5) - 1
        xapp = np.hstack((np.random.randn(2 * nbapp, 2) + np.hstack((3 * A, A * (yapp * 1.875 + 1.125))), 20 * np.random.randn(nbapp * 2, 8)))
        nbtest = round(nbtest / 2)
        ytest = np.vstack((np.ones((nbtest, 1)), -1 * np.ones((nbtest, 1))))
        A = 2 * (np.random.rand(2 * nbtest, 1) < 0.5) - 1
        xtest = np.hstack((np.random.randn(2 * nbtest, 2) + np.hstack((3 * A, A * (ytest * 1.875 + 1.125))), 20 * np.random.randn(nbtest * 2, 8)))

    elif n.lower()  == 'gaussianmultires':
        n = nbapp // 12

        xm1 = 1
        # The Big One
        mean1 = [-xm1, -xm1]
        mean2 = [xm1, xm1]
        sigma1 = 0.8
        sigma2 = 0.3
        xapp1 = np.ones((2 * n, 1)) * mean1 + sigma1 * np.random.randn(2 * n, 2)
        xapp2 = np.ones((2 * n, 1)) * mean2 + sigma1 * np.random.randn(2 * n, 2)
        yapp1 = np.ones((n, 1))
        yapp2 = -np.ones((n, 1))
        # The small ones
        mean1 = [-2 * xm1, 0]
        mean2 = [0, -2 * xm1]
        mean3 = [-2 * xm1, -2 * xm1]
        mean4 = [-2 * xm1, -2 * xm1]

        xapp11 = np.ones((n, 1)) * mean1 + sigma2 * np.random.randn(n, 2)
        xapp12 = np.ones((n, 1)) * mean2 + sigma2 * np.random.randn(n, 2)
        xapp13 = np.ones((n, 1)) * mean3 + sigma2 * np.random.randn(n, 2)
        xapp14 = np.ones((n, 1)) * mean4 + sigma2 * np.random.randn(n, 2)
        # The small ones
        mean1 = [2 * xm1, 0]
        mean2 = [0, 2 * xm1]
        mean3 = [2 * xm1, 2 * xm1]
        mean4 = [2 * xm1, 2 * xm1]

        xapp21 = np.ones((n, 1)) * mean1 + sigma2 * np.random.randn(n, 2)
        xapp22 = np.ones((n, 1)) * mean2 + sigma2 * np.random.randn(n, 2)
        xapp23 = np.ones((n, 1)) * mean3 + sigma2 * np.random.randn(n, 2)
        xapp24 = np.ones((n, 1)) * mean4 + sigma2 * np.random.randn(n, 2)
        xapp = np.concatenate((xapp1, xapp21, xapp22, xapp23, xapp24, xapp2, xapp11, xapp12, xapp13, xapp14), axis=0)
        yapp = np.concatenate((yapp1, -yapp1, yapp2, -yapp2, yapp1, -yapp1), axis=0)

        xtest = np.array([])
        ytest = np.array([])
        

    elif n.lower()  ==  'carpette':
        xapp = 2 * np.random.rand(nbapp, 2) - 1
        xtest = 2 * np.random.rand(nbtest, 2) - 1
        xapp = xapp[np.sum(xapp ** 2, axis=1) <= 0.64, :]
        xtest = xtest[np.sum(xtest ** 2, axis=1) <= 0.64, :]

        yapp = carpette(xapp)
        ytest = carpette(xtest)

    elif n.lower()  ==  'gaussianmultires2':        
        n = nbapp // 10
        sigma = 1.5
        
        xappaux1 = sigma * np.random.randn(10 * nbapp, 2)
        rayon2 = np.sum(xappaux1**2, axis=1)
        ind = np.where((rayon2 > 0.14) & (rayon2 < 0.64))[0]
        xappaux1 = xappaux1[ind[:n], :]
        
        xappaux2 = sigma * np.random.randn(10 * nbapp, 2)
        rayon2 = np.sum(xappaux2**2, axis=1)
        ind = np.where(rayon2 < 0.14)[0]
        xappaux2 = xappaux2[ind[:n], :]
        
        xappaux3 = sigma * np.random.randn(10 * nbapp, 2)
        rayon2 = np.sum(xappaux3**2, axis=1)
        ind = np.where(rayon2 > 0.64)[0]
        xappaux3 = xappaux3[ind[:4*n], :]
        
        xapp = np.concatenate([xappaux2, xappaux3, xappaux1], axis=0)
        yapp = np.concatenate([np.ones(5*n), -np.ones(n)], axis=0)
        
        mean1 = [-4, 0]
        mean2 = [0, 4]
        mean3 = [4, 0]
        mean4 = [0, -4]
        sigma1 = 0.5
        
        xapp1 = np.ones((n, 1)) * mean1 + sigma1 * np.random.randn(n, 2)
        xapp2 = np.ones((n, 1)) * mean2 + sigma1 * np.random.randn(n, 2)
        xapp3 = np.ones((n, 1)) * mean3 + sigma1 * np.random.randn(n, 2)
        xapp4 = np.ones((n, 1)) * mean4 + sigma1 * np.random.randn(n, 2)
        
        xapp = np.concatenate([xapp, xapp1, xapp2, xapp3, xapp4], axis=0)
        yapp = np.concatenate([yapp, -np.ones(4*n)], axis=0)



    return xapp, yapp, xtest, ytest, xtest1, xtest2
