import pylab as pl
from GenericFVUtils import *

def maxAbsEig(self, U, dx, dy):
    return max(np.max(np.abs(self.params.a))/dx,np.max(np.abs(self.params.b))/dy)

def numFluxX_upwind(self, U, dt, dx):
    mask = self.params.a < 0.
    Fu = np.empty_like(U[0].uW)
    Fu[-mask] = self.params.a[-mask]*U[0].uW[-mask]
    Fu[ mask] = self.params.a[ mask]*U[0].uE[ mask]
    return [Fu]

def numFluxY_upwind(self, U, dt, dx):
    mask = self.params.b < 0.
    Fu = np.empty_like(U[0].uS)
    Fu[-mask] = self.params.b[-mask]*U[0].uS[-mask]
    Fu[ mask] = self.params.b[ mask]*U[0].uN[ mask]
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
    hcl = HyperbolicConsLaw(order, limiter)

#set numerical Flux
    hcl.setNumericalFluxFuns(numFluxX_upwind, numFluxY_upwind, maxAbsEig)

# set boundary conditions
    boundaryCondFunN = None
    boundaryCondFunS = None
    boundaryCondFunW = None
    boundaryCondFunE = None
    hcl.setBoundaryCond(boundaryCondFunE, boundaryCondFunW, boundaryCondFunN, boundaryCondFunS)

# set initial state
    xCc = np.linspace(0.+.5/nx,1.-.5/nx,nx) # cell centers
    yCc = np.linspace(0.+.5/ny,1.-.5/ny,ny) # cell centers
    xv, yv = np.meshgrid(xCc, yCc)
    uinit = np.zeros((ny, nx))
    uinit[np.sqrt((xv-.5)**2 + (yv-.5)**2)<.25] = 1.
    hcl.setU([uinit], ny, nx, xCc, yCc)

# set flux parameters
    a_ = .5*np.ones((ny,nx+1))
    b_ = .5*np.ones((ny+1,nx))
    hcl.setFluxParams(a = a_, b = b_)

# apply explicit time stepping
    t = 0.
    while t<Tmax:
        t = hcl.timeStepExplicit(t, Tmax)

#plot result
    pl.figure(3)
    pl.title('linear 2d')
    pl.ion()
    pl.pcolor(xv, yv, hcl.getU(0), cmap=pl.gray())

    return hcl

