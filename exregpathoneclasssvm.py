import numpy as np
import matplotlib.pyplot as plt
from datasets import datasets
from regpathsvmoneclass import regpathsvmoneclass
from svmval import svmval
from TransformPathFromNu import TransformPathFromNu
from matplotlib.animation import FFMpegWriter

# Clear variables and close all figures
plt.close('all')
np.random.seed(43)

# Generate dataset
n = 100
sigma = 0.3
xapp, yapp, *other = datasets('gaussian', n, 0, sigma)
xapp = xapp[yapp == -1, :]
nbtrain = xapp.shape[0]

# Generate test data
xtest1, xtest2 = np.meshgrid(np.arange(-1, 1.01, 0.01)*3.5, np.arange(-1, 1.01, 0.01)*3)
xtest = np.vstack((xtest1.ravel(), xtest2.ravel())).T
nn = len(xtest1)
kernel = 'gaussian'
kerneloption = np.array([[1]])
verbose = 1

# Compute regularization path
alphamat, alpha0vec, lambdavec, event = regpathsvmoneclass(xapp, kernel, kerneloption, verbose)
Nbapp = xapp.shape[0]
N = 30
nuvec = np.linspace(np.min(lambdavec)/Nbapp, np.max(lambdavec)/Nbapp, N)[::-1]
nuvec = np.linspace(0.05, 0.95, N)[::-1]
alphamat, alpha0vec, lambdavec, nuvec = TransformPathFromNu(alphamat, alpha0vec, lambdavec, nuvec, Nbapp)

# Plot decision boundary for selected values of lambda
xsup = xapp
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
ax1.plot(xapp[:, 0], xapp[:, 1], '+r')
ax1.axis([-1.5, 0.5, -1.5, 1])
for i in range(0, len(lambdavec), 8):
    w = alphamat[:, i]
    w0 = -alpha0vec[:, i]
    ypred = svmval(xtest, xsup, w, w0, kernel, kerneloption, 1)/lambdavec[i]
    ypred = ypred.reshape(nn, nn)
    cs = ax1.contour(xtest1, xtest2, ypred, [0], colors='k', linewidths=2)
    plt.clabel(cs, inline=1, fontsize=10)

# Plot regularization path
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
ax2.plot(lambdavec, alphamat.T, linewidth=2)
ax2.axis([np.min(lambdavec), np.max(lambdavec), 0, 1])
ax2.set_xlabel('$\lambda$', fontsize=16)
ax2.set_ylabel('$\\alpha$', fontsize=16)

plt.show()

#Create a VideoWriter object
writerObj = FFMpegWriter(fps=2)
writerObj.setup('exampleoneclassalpha.avi')

#Loop over the frames and write each one to the video
for i in range(len(lambdavec)-2):
    fig2 = plt.figure(2)
    fig2.clf()
    plt.plot(1./lambdavec[0:i+1], alphamat[:,0:i+1].T, linewidth=2)
    plt.xlabel('1/(\nu . n)')
    plt.xscale('log')
    fig2.set_facecolor('white')
    plt.axis([0, 0.3, 0, 1])
    F = plt.gcf()
    writerObj.grab_frame()

#Close the video writer
writerObj.finish()

#Create a VideoWriter object
v = FFMpegWriter(fps=2)
v.setup('exampleoneclass.avi')

xsup = xapp
for i in range(len(lambdavec)):
    w = alphamat[:,i]
    w0 = -alpha0vec[:,i]
    ypred = svmval(xtest, xsup, w, w0, kernel, kerneloption, 1) / lambdavec[i]
    ypred = ypred.reshape(nn, nn)
    fig1 = plt.figure(1)
    fig1.clf()
    cc, hh = plt.contour(xtest1, xtest2, ypred, [0, 0], colors='k')
    plt.hold(True)
    plt.setp(hh, linewidth=2)
    fig1.set_facecolor('white')
    h1 = plt.plot(xapp[:,0], xapp[:,1], '+r')
    plt.axis([-3, 3, -3, 3])
    F = plt.gcf()
    v.grab_frame()

#Close the video writer
v.finish()