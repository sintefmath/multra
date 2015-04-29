import pylab as pl
from GenericFVUtils import *

def maxAbsEig(self, U, dx, dy):
    return np.max(np.abs(self.a))/dx

def FluxX(self, u, m):
    return self.a[m]*u[m]

def numFluxX_upwind(self, U, dt, dx):
    mask = self.a < 0.
    Fu = np.empty_like(U[0].uW)
    Fu[-mask] = self.a[-mask]*U[0].uW[-mask]
    Fu[ mask] = self.a[ mask]*U[0].uE[ mask]
    return [Fu]

def boundaryCondFunE(t, dx, y):
    return [0.]

def boundaryCondFunW(t, dx, y):
# square pulse
    u = 0.
    if (t<0.4):
        u = 0.25
    return [u]

def linear(nx=1000, Tmax=1., order=1, limiter='minmod'):

# generate instance of class
    hcl = HyperbolicConsLaw(order, limiter)

#set numerical Flux
    numFluxX = numFluxX_upwind
    numFluxY = None
    hcl.setNumericalFluxFuns(numFluxX, numFluxY, maxAbsEig)

# set boundary conditions
    boundaryCondFunN = None
    boundaryCondFunS = None
    hcl.setBoundaryCond(boundaryCondFunE, boundaryCondFunW, boundaryCondFunN, boundaryCondFunS)

# set initial state
    xCc = np.linspace(0.+.5/nx,1.-.5/nx,nx) # cell centers
    yCc = None
    ny = None
    uinit = np.zeros((nx))
    #uinit[order:-order] = np.sin(2*np.pi*xCc)
    hcl.setU([uinit], nx, ny, xCc, yCc)

# set flux parameters
    xCi = np.linspace(0,1,nx+1) # cell interfaces
    a_ = -xCi + .5
    hcl.setFluxParams(a = a_)

# apply explicit time stepping
    t = 0.
    while t<Tmax:
        t = hcl.timeStepExplicit(t, Tmax)

#plot result
    pl.ion()
    pl.plot(hcl.xCc,hcl.getU(0))
    pl.plot(xCi,a_,'k:')

    return hcl
