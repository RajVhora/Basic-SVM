import numpy as np

def TransformPathFromNu(alphamat, alpha0vec, lambdavec, nuvec, Nbapp):
    N = len(nuvec)
    # Nbapp = xapp.shape[0]
    # nuvec = np.flipud(np.linspace(np.min(lambdavec)/Nbapp, np.max(lambdavec)/Nbapp, N))
    lambdatrue = nuvec * Nbapp
    seuil = 1e-6
    newalphamat = np.zeros((alphamat.shape[0], N))
    newalpha0vec = np.zeros(N)

    for i in range(N):
        lambdacurrent = lambdatrue[i]
        ind = np.where(abs(lambdavec - lambdacurrent) <= seuil * Nbapp)[0]
        
        if len(ind) > 1:
            ind = ind[0]
        if len(ind) == 0:
            mini = np.min(abs(lambdavec - lambdacurrent))
            ind = np.argmin(abs(lambdavec - lambdacurrent))
            if lambdavec[ind] < lambdacurrent:
                lambdamoins = lambdavec[ind]
                lambdaplus = lambdavec[ind - 1]
                plus = ind - 1
                moins = ind
            else:
                if ind + 1 <= len(lambdavec) - 1:
                    lambdamoins = lambdavec[ind + 1]
                    lambdaplus = lambdavec[ind]
                    plus = ind
                    moins = ind + 1
                else:
                    moins = ind
                    plus = ind
                    lambdamoins = lambdacurrent
                    lambdaplus = 0
            newalphamat[:, i] = alphamat[:, moins] + (lambdacurrent - lambdamoins) * (alphamat[:, plus] - alphamat[:, moins]) / (lambdaplus - lambdamoins)
            newalpha0vec[i] = alpha0vec[moins] + (lambdacurrent - lambdamoins) * (alpha0vec[plus] - alpha0vec[moins]) / (lambdaplus - lambdamoins)
        else:
            newalphamat[:, i] = alphamat[:, ind]
            newalpha0vec[i] = alpha0vec[ind]

    return newalphamat, newalpha0vec, lambdatrue, nuvec
