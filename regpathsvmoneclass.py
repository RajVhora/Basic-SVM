from normalizekernel import normalizekernel
from monqp import monqp
import matplotlib.pyplot as plt
import warnings
from svmval import svmval


def regpathsvmoneclass(xapp, kernel, kerneloption, verbose):
    nuinit = 0.999
    lambdaseuil = 1.1
    lambdaseuil = int(xapp.shape[0] * 0.01)
    epsilon = 1e-8
    DOQP = 1
    lambd = 1e-8
    chouia = 1e-3

    event = []
    lambdavec = []
    alphamat = []
    alpha0vec = []

    import numpy as np

    # Calculate number of samples and dimensionality of features
    nbtrain, dim = xapp.shape

    # Initialize regularization parameter and kernel matrix
    lambda_val = nuinit * nbtrain
    Kapp = normalizekernel(xapp, kernel, kerneloption) + epsilon * np.eye(nbtrain)

    # Initialization
    if DOQP:
        # Initialize parameters for quadratic program
        c = np.zeros((nbtrain, 1))
        A = np.ones((nbtrain, 1))
        b = lambda_val
        C = 1
        
        # Generate initial alpha values randomly
        indrand = np.random.permutation(nbtrain)
        Elambda = int(np.floor(lambda_val))
        alphainit = np.zeros((nbtrain, 1))
        alphainit[indrand[:Elambda], 0] = 1
        alphainit[indrand[Elambda], 0] = lambda_val - Elambda
        
        # Solve quadratic program
        alphaaux, multiplier, pos = monqp(Kapp / lambda_val, c, A, b, C, lambd, verbose, None, None, alphainit)
        
        # Check if any alpha values are outside of [0, 1] range
        if np.sum((alphaaux > 1) | (alphaaux < 0)) > 0:
            print('Error Init')
        
        # Update alpha, alpha0, and fx values
        alpha = np.zeros((nbtrain, 1))
        alpha[pos] = alphaaux
        alpha0 = -lambda_val * multiplier
        fx = (Kapp @ alpha - alpha0) / lambda_val
        
        # Store results
        lambdavec = [lambda_val]
        alphamat = alpha
        alpha0vec = [alpha0]
        elbow = np.where(np.abs(fx) < lambd)[0]
        left = np.where(fx < -lambd)[0]
        right = np.where(fx > lambd)[0]
    else:
        # Calculate initial alpha values and alpha0 for Parzen windows
        alpha = np.ones((nbtrain, 1))
        alpha0 = np.sqrt(alpha.T @ Kapp @ alpha)
        fx = (Kapp @ alpha - alpha0) / nbtrain
        
        # Find index of maximum fx value below 0 and update alpha accordingly
        indaux = np.where(fx < 0)[0]
        indaux1 = np.argmax(fx[indaux])
        indaux1 = np.where(fx == fx[indaux[indaux1]])[0][0]
        if len(indaux1) > 1:
            print('Doublons')
            indaux1 = indaux1[0]
        alpha[indaux[indaux1]] = lambda_val - np.floor(lambda_val)
        alpha0 = Kapp[indaux[indaux1[0]], :] @ alpha
        fx = (Kapp @ alpha - alpha0) / lambda_val
        
        # Store results
        lambdavec = [lambda_val]
        alphamat = alpha
        alpha0vec = [alpha0]
        elbow = [indaux[indaux1]]
        right = []
        left = np.setdiff1d(np.arange(nbtrain), elbow)

    #Main

    while lambda_val > lambdaseuil and lambda_val > 1:
        if verbose == 1:
            print(f"lambda={lambda_val:.3f}", end="\r")

        nbelbow = len(elbow)
        if nbelbow != 0:
            Una = np.concatenate((np.zeros((nbelbow, 1)), np.ones((1, 1))))
            A = np.concatenate((np.concatenate((Kapp[np.ix_(elbow, elbow)], -np.ones((nbelbow, 1))), axis=1), 
                                np.concatenate((np.ones((1, nbelbow)), np.zeros((1, 1))), axis=1)), 
                            axis=0)
            ba = np.linalg.solve(A, Una)
            balpha = ba[:-1]
            bo = ba[-1]
            
            # case elbow to 0 or 1
            lambda1 = lambda_val + (np.ones((nbelbow, 1))-alpha[elbow])/balpha
            lambda2 = lambda_val + (np.zeros((nbelbow, 1))-alpha[elbow])/balpha
            
            # case L and R
            hellxi = np.dot(Kapp[:, elbow], balpha) - bo
            lambda3 = lambda_val*(fx[left]-hellxi[left])/(-hellxi[left])
            lambda4 = lambda_val*(fx[right]-hellxi[right])/(-hellxi[right])
            
            ind1 = np.where((lambda1+epsilon) < lambda_val)[0]
            ind2 = np.where((lambda2+epsilon) < lambda_val)[0]
            ind3 = np.where((lambda3+epsilon) < lambda_val)[0]
            ind4 = np.where((lambda4+epsilon) < lambda_val)[0]
            
            lambdanew = np.max(np.concatenate((lambda1[ind1], lambda2[ind2], lambda3[ind3], lambda4[ind4])))
            
            if lambdanew == 0:
                print('Exit due to no new lambda ...')
                break    
            
            fx = lambda_val/lambdanew*(fx-hellxi) + hellxi
            alpha[elbow] = alpha[elbow] - (lambda_val-lambdanew)*balpha
            alpha0 = alpha0 - (lambda_val-lambdanew)*bo
            lambda_val = lambdanew
            
            # mise à jour des ensembles
            if lambdanew in lambda1: # event 1 elbow to left
                ind = np.where(np.abs(lambda1-lambdanew) < epsilon)[0]
                left = np.concatenate((left, elbow[ind]))
                elbow = np.delete(elbow, ind)
                event = np.concatenate((event, [1]*len(ind)))
                
            if lambdanew in lambda2: # event 2 elbow to right
                ind = np.where(np.abs(lambda2-lambdanew) < epsilon)[0]
                right = np.concatenate((right, elbow[ind]))
                elbow = np.delete(elbow, ind)
                event = np.concatenate((event, [2]*len(ind)))
            
            if lambdanew in lambda3: # event 3 left to elbow
                ind = np.where(np.abs(lambda3-lambdanew) < epsilon)[0]
                elbow = np.concatenate((elbow, left[ind]))
                left = np.delete(left, ind)
                event = np.concatenate((event, [3]*len(ind)))
                
            if lambdanew in lambda4: # event 4 right to elbow
                ind = np.where(np.abs(lambda4-lambdanew) < epsilon)[0]
                elbow = np.concatenate((elbow, right[ind]))
                left = np.delete(right, ind)
                event = np.concatenate((event, [4]*len(ind)))

        else:

            # if elbow is empty then redo initialisation
            lambda_val = lambda_val - chouia

            #--------------------------------------------------
            # 
            #  initialisation with QP
            if DOQP:
                c = np.zeros(nbtrain)
                A = np.ones(nbtrain)
                b = lambda_val
                C = 1
                alphainit = alpha
                indaux = np.where(fx < 0)[0]
                aux, indaux1 = np.max(fx[indaux]), np.argmax(fx[indaux])
                alphainit[indaux[indaux1]] = 1 - chouia

                alphaaux, multiplier, pos = monqp(Kapp/lambda_val, c, A, b, C, epsilon, verbose, [], [], alphainit)
                if np.sum(alphaaux > 1) > 0:
                    print('*')
                    break

                alpha = np.zeros(nbtrain)
                alpha[pos] = alphaaux
                alpha0 = -lambda_val * multiplier
                fx = (Kapp @ alpha - alpha0) / lambda_val
                elbow = np.where(np.abs(fx) < epsilon)[0]
                left = np.where(fx < -epsilon)[0]
                right = np.where(fx > epsilon)[0]
            else:
                # alternative heuristics
                alphainit = alpha
                indaux = np.where(fx < 0)[0]
                aux, indaux1 = np.max(fx[indaux]), np.argmax(fx[indaux])
                indaux1 = np.where(fx[indaux] == aux)[0][0]  # see if there is any doublons
                if len(indaux1) > 1:
                    warnings.warn('doublons')
                    indaux1 = indaux1[0]

                alphainit[indaux[indaux1]] = 1 - chouia
                alpha = alphainit
                alpha0 = Kapp[indaux[indaux1], :] @ alpha
                if alpha0.size == 0:
                    alpha0 = 0
                fx = (Kapp @ alpha - alpha0) / lambda_val
                elbow = np.where(np.abs(fx) < epsilon)[0]
                left = np.where(fx < -epsilon)[0]
                right = np.where(fx > epsilon)[0]
                event = np.append(event, 5)

        alpha[right] = 0
        alpha[left] = 1
        indaux = np.where(fx[right] < epsilon)
        fx[right[indaux]] = epsilon

        lambdavec.append(lambda_val)
        alphamat.append(alpha)
        alpha0vec.append(alpha0)

        elbow = np.sort(elbow)
        right = np.sort(right)
        left = np.sort(left)
        if verbose > 2:
            xsup = xapp
            w = alpha
            w0 = -alpha0
            ypred = svmval(xtest, xsup, w, w0, kernel, kerneloption, 1) / lambda_val
            ypred = ypred.reshape(nn, nn)
            plt.figure(1)
            plt.subplot(2, 1, 1)
            # plt.contourf(xtest1,xtest2,ypred,50);shading flat;
            plt.contour(xtest1, xtest2, ypred, [0, 0], colors='k', linewidths=2)
            plt.plot(xapp[:, 0], xapp[:, 1], '+r', linewidth=2)
            plt.plot(xapp[elbow, 0], xapp[elbow, 1], 'dg')
            plt.subplot(2, 1, 2)
            plt.plot(alphamat.T)
            plt.draw()

