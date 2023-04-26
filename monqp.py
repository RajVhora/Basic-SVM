import numpy as np
from cout import cout
def monqp(H, c, A, b, C, l=0, verbose=0, X=None, ps=None, xinit=None):
    n, d = H.shape
    nl, nc = c.shape
    nlc, ncc = A.shape
    nlb, ncb = (1,1) if isinstance(b, float) else b.shape
    Cs1, Cs2 = (1,1) if isinstance(C, float) or isinstance(C,int) else C.shape
    if d != n:
        raise ValueError('H must be a square matrix n by n')

    if nl != n:
        raise ValueError('H and c must have the same number of row')

    if nlc != n:
        raise ValueError('H and A must have the same number of row')

    if nc != 1:
        raise ValueError('c must be a row vector')

    if ncb != 1:
        raise ValueError('b must be a row vector')

    if ncc != nlb:
        raise ValueError('A\' and b must have the same number of row')

    if Cs1 != nl:
        C = C * np.ones((nl, 1))

    if xinit is None:
        xinit = []

    if l is None:
        l = 0

    if verbose is None:
        verbose = 0

    fid = 1 # default value, current matlab window

    OO = np.zeros((ncc,1))
    H = H + l * np.eye(len(H)) # preconditioning

    xnew = -1 * np.ones(C.shape)
    ness = 0
    ind = 1

    if xinit.size == 0:
        while (np.sum(xnew < 0) > 0 or np.sum(xnew > C[ind]) > 0) and ness < 100:
            ind = np.sort(np.random.permutation(n)[:ncc])
            aux = np.vstack(
            [
            np.hstack([H[ind][:, ind], A[ind, :]]),
            np.hstack([A[ind, :].T, OO]),
            ]
            )
            aux = aux + l * np.eye(aux.shape[0])
            if np.linalg.cond(aux) > 1e-12:
                newsol = np.linalg.solve(aux, np.hstack([c[ind], b]))
                xnew = newsol[: len(ind)]
                ness += 1
            else:
                ness = 101
                break

        if ness < 100:
            x = xnew
            lambda_ = newsol[len(ind) :]
        else:
            ind = [0]
            x = C[ind] / 2 * np.ones((ind.shape[0], 1))
            lambda_ = np.ones((ncc, 1))

        indsuptot = []
    else:
        indsuptot = np.where(xinit == C)[0]
        indsup = indsuptot
        ind = np.where((xinit > 0) & (xinit != C))[0]
        x = xinit[ind]
        lambda_ = np.ones((ncc, 1))

    if np.sum(A == 0):
        ncc = 0

    try:
        U = np.linalg.cholesky(H)
        testchol = 0
    except:
        testchol = 1

    
    firstchol = 1


    #Main Loop

    Jold = 10000000000000000000  
    # C = Jold # for the cost function
    if verbose != 0:
        print('      Cost     Delta Cost  #support  #up saturate')
        nbverbose = 0

    nbiter = 0
    STOP = 0
    nbitermax = 20 * n

    while STOP != 1:

        nbiter += 1
        indd = np.zeros((n,1))
        for i in ind:
            indd[i,0] = i
        nsup = len(ind)
        indd[indsuptot] = -1

        if verbose != 0:
            J, yx = cout(H, x, b, C, indd, c, A, b, lambda_)
            nbverbose += 1

            if nbverbose == 20:
                print('      Cost     Delta Cost  #support  #up saturate')
                nbverbose = 0

            if Jold == 0:
                print("| %11.4e | %8.4f | %6.0f | %6.0f |" % (J, (Jold - J), nsup, len(indsuptot)))
            elif (Jold - J) > 0:
                print("| %11.4e | %8.4f | %6.0f | %6.0f |" % (J, min((Jold - J) / abs(Jold), 99.9999), nsup, len(indsuptot)))
            else:
                print("| %11.4e | %8.4f | %6.0f | %6.0f | bad move " % (J, max((Jold - J) / abs(Jold), -99.9999), nsup, len(indsuptot)))
            Jold = J

        ce = c[ind]
        be = b
        if len(indsuptot)!=0:
            Cmat = np.ones((len(ind), 1)) * C[indsuptot].T
            #if ce.shape != np.sum(Cmat * H[ind][:, indsuptot], axis=1).shape:
            #    keyboard
            ce = ce - np.sum(np.multiply(Cmat, H[ind][:, indsuptot]), axis=1)[0]
            Cmat = C[indsuptot] * np.ones((1, A.shape[1]))
            be = be - np.sum(Cmat * A[indsuptot, :], axis=0).T

        At = A[ind, :].T
        Ae = A[ind, :]

        if testchol == 0:
            #auxH = H[ind,ind]
            auxH = H[ind,:][:,ind]
            # reshape auxH to 2d by adding a dimension of size 1
            #auxH = auxH.reshape(auxH.shape[0], 1)
            try:
                U = np.linalg.cholesky(auxH)
                testchol = 0
            except:
                testchol = 1
        
            M = At @ (np.linalg.solve(U.T, np.linalg.solve(U, Ae)))
            d = np.linalg.solve(U.T, np.linalg.solve(U, ce))
            d = At @ d - be

            if np.linalg.cond(M) < l:
                M += l * np.eye(M.shape[0])
            lambda_ = np.linalg.solve(M, d)

            xnew = np.linalg.solve(auxH, ce - Ae @ lambda_)

        else:
            auxM = At @ np.linalg.solve(auxH, Ae)
            M = auxM.T @ auxM
            d = np.linalg.solve(auxH, ce)
            d = At @ d - be

            if np.linalg.cond(M) < l:
                M += l * np.eye(M.shape[0])
            lambda_ = np.linalg.solve(M, d)
            xnew = np.linalg.solve(auxH, ce - Ae @ lambda_)

        minxnew = np.min(xnew)
        minpos = np.argmin(xnew)

        if (np.sum(xnew < 0) > 0 or np.sum(xnew > C[ind]) > 0) and len(ind) > ncc:
            d = (xnew - x) + l
            indad = np.where(xnew < 0)[0]
            indsup = np.where(xnew > C[ind])[0]
            tI, indmin = np.min(-x[indad] / d[indad]), np.argmin(-x[indad] / d[indad])
            tS, indS = np.min((C[ind[indsup]] - x[indsup]) / d[indsup]), np.argmin((C[ind[indsup]] - x[indsup]) / d[indsup])
            if np.isnan(tI):
                tI = tS + 1
            if np.isnan(tS):
                tS = tI + 1
            t = min(tI, tS)
            x = x + t * d

            if t == tI:
                varcholupdate = ind[indad[indmin]]
                indexcholupdate = indad[indmin]
                directioncholupdate = -1
                ind = np.delete(ind, indad[indmin])
                x = np.delete(x, indad[indmin])
            else:
                indexcholupdate = indsup[indS]
                varcholupdate = ind[indsup[indS]]
                directioncholupdate = -1
                indsuptot = np.concatenate((indsuptot, np.array([ind[indsup[indS]]])))
                ind = np.delete(ind, indsup[indS])
                x = np.delete(x, indsup[indS])
        else:
            xt = np.zeros((n,1)) # keyboard
            xt[ind] = xnew # keyboard
            xt[indsuptot] = C[indsuptot]
            indold = ind ## 03/01/2002

            mu = H @ xt - c + A @ lambda_ # calcul des multiplicateurs de lagrange associées aux contraintes

            indsat = np.arange(0,n) # on ne regarde que les contraintes saturées
            #indsat = np.setdiff1d(indsat, np.concatenate((ind, indsuptot)))
            indsat = np.empty((0,0))
 
            mm, mpos = (None,None) if indsat.shape[0]==0 else (np.min(mu[indsat]), np.argmin(mu[indsat]))
            mmS, mposS = np.min(-mu[indsuptot]), np.argmin(-mu[indsuptot])

            
            if (((mm < -np.sqrt(2.2204e-16)) and (mm is not None)) or ((mmS < -np.sqrt(2.2204e-16)) and (mmS is not None))) and (nbiter < nbitermax):
                if (len(indsuptot) == 0) or (mm < mmS):
                    ind = np.sort(np.concatenate((ind, [indsat[mpos]]))) # il faut rajouter une variable
                    x = xt[ind]
                    indexcholupdate = np.where(ind == indsat[mpos])[0][0]
                    varcholupdate = indsat[mpos]
                    directioncholupdate = 1 # remove
                else:
                    ind = np.sort(np.concatenate((ind, [indsuptot[mposS]]))) # on elimine la contrainte sup si necessaire
                    x = xt[ind] # on elimine une contrainte de type x=C
                    indexcholupdate = np.where(ind == indsuptot[mposS])[0][0]
                    varcholupdate = indsuptot[mposS]
                    indsuptot = np.delete(indsuptot, mposS)
                    directioncholupdate = 1 # remove
            else:
                STOP = 1
                pos = np.sort(np.concatenate((ind, indsuptot)))
                xt = np.zeros((n,1))
                xt[ind] = xnew
                xt[indsuptot] = C[indsuptot]
                indout = np.arange(0,n)
                indout = np.setdiff1d(indout, pos)
                xnew = xt[indout]
    
    return xnew, lambda_, pos