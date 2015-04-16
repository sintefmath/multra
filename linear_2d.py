import pylab as pl
from GenericFVUtils import *

a = None
b = None

def maxAbsEig(U, dx, dy):
    global a
    global b
    return max(np.max(np.abs(a))/dx,np.max(np.abs(b))/dy)

def FluxX(u, m):
    global a
    return a[m]*u[m]

def FluxY(u, m):
    global b
    return b[m]*u[m]

def numFluxX_upwind(U, dt, dx):
    global a
    mask = a < 0.
    Fu = np.empty_like(U[0].uW)
    Fu[-mask] = FluxX(U[0].uW,-mask)
    Fu[mask] = FluxX(U[0].uE,mask)
    return [Fu]

def numFluxY_upwind(U, dt, dx):
    global b
    mask = b < 0.
    Fu = np.empty_like(U[0].uS)
    Fu[-mask] = FluxY(U[0].uS,-mask)
    Fu[mask] = FluxY(U[0].uN,mask)
    return [Fu]

def boundaryCondFunE(t, dx, y):
    return [0.]
def boundaryCondFunW(t, dx, y):
# square pulse
    u = 0.
    if (t<0.4):
        u = 0.25
    return [u]

def linear(nx=100, ny=100 ,Tmax=1., order=1, limiter='minmod'):

# generate instance of class
    hcl = HyperbolicConsLaw()

    boundaryCondFunN = None
    boundaryCondFunS = None
    boundaryCondFunW = None
    boundaryCondFunE = None

# set functions for max absolute eigenvalue, fluxes, boundaryconditions, order, and limiter
    hcl.setFuns(maxAbsEig, numFluxX_upwind, numFluxY_upwind, boundaryCondFunE, boundaryCondFunW, boundaryCondFunN, boundaryCondFunS, order, limiter)

    xCc = np.linspace(0.+.5/nx,1.-.5/nx,nx) # cell centers
    yCc = np.linspace(0.+.5/ny,1.-.5/ny,ny) # cell centers

    xv, yv = np.meshgrid(xCc, yCc)

    uinit = np.zeros((ny, nx))
    uinit[np.sqrt((xv-.5)**2 + (yv-.5)**2)<.25] = 1.

# set initial state
    hcl.setU([uinit], ny, nx, xCc, yCc)

    global a
    a = .5*np.ones((ny,nx+1))
    global b
    b = .5*np.ones((ny+1,nx))

    t = 0.
    while t<Tmax:
# apply explicit time stepping
        t = hcl.timeStepExplicit(t, Tmax)

#plot result
    pl.ion()
    pl.pcolor(xv, yv, hcl.U[0].u[order:-order,order:-order], cmap=pl.gray())

    xCi = np.linspace(0,1,nx+1) # cell interfaces
    yCi = np.linspace(0,1,ny+1) # cell interfaces

    return xCi, yCi, hcl.U

