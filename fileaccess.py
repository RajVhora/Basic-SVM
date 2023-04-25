import numpy as np

def fileaccess(datafile, Indice, Dimension, fid=None, Format=16, Offset=0, CaracEntreExemple=2):
    if fid is None:
        fid = open(datafile, 'rb')

    N = len(Indice)
    X = np.zeros((N, Dimension))
    for i in range(N):
        if Dimension != 1:
            fid.seek((Dimension*Format + CaracEntreExemple)*(Indice[i]-1) - Offset)
            X[i,:] = np.fromfile(fid, dtype=np.float32, count=Dimension)
        else:
            fid.seek((Format)*(Indice[i]-1) - Offset)
            X[i,:] = np.fromfile(fid, dtype=np.float32, count=1)
        
    return X
