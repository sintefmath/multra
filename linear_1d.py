import pylab as pl
from GenericFVUtils import *

a = None

def maxAbsEig(U, dx, dy):
    global a
    return np.max(np.abs(a))/dx

def FluxX(u, m):
    global a
    return a[m]*u[m]

def numFluxX_upwind(U, dt, dx):
    global a
    mask = a < 0.
    Fu = np.empty_like(U[0].uW)
    Fu[-mask] = FluxX(U[0].uW,-mask)
    Fu[mask] = FluxX(U[0].uE,mask)
    return [Fu]

def boundaryCondFunE(t, dx, y):
    return [0.]

def boundaryCondFunW(t, dx, y):
    u = 0.
    x0 = 0.4
    if (t<x0):
        u = 0.25
    return [u]

def linear(nx=1000, Tmax=1., order=1, limiter='minmod'):

# generate instance of class
    hcl = HyperbolicConsLaw()

    numFluxX = numFluxX_upwind
    numFluxY = None

    boundaryCondFunN = None
    boundaryCondFunS = None

# set functions for max absolute eigenvalue, fluxes, boundaryconditions, order, and limiter
    hcl.setFuns(maxAbsEig, numFluxX, numFluxY, boundaryCondFunE, boundaryCondFunW, boundaryCondFunN, boundaryCondFunS, order, limiter)

    xCc = np.linspace(0.+.5/nx,1.-.5/nx,nx) # cell centers

    ny = None
    yCc = None

    uinit = np.zeros((nx))
    #uinit[order:-order] = np.sin(2*np.pi*xCc)

# set initial state
    hcl.setU([uinit], nx, ny, xCc, yCc)

#
    xCi = np.linspace(0,1,nx+1) # cell interfaces
    global a
    a = -xCi + .5

    t = 0.
    while t<Tmax:
# apply explicit time stepping
        t = hcl.timeStepExplicit(t, Tmax)

#plot result
    pl.ion()
    pl.plot(xCc,hcl.U[0].u[order:-order])
    pl.plot(xCi,a,'k:')

    return xCi, hcl.U
