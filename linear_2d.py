import pylab as pl
from GenericFVUtils import *

def maxAbsEig(hcl):
    return max(np.max(np.abs(hcl.params.a))/hcl.dx,np.max(np.abs(hcl.params.b))/hcl.dy)

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

def initialCondFun(xv, yv):
    u = np.zeros_like(xv)
    u[np.sqrt((xv-.25)**2 + (yv-.25)**2)<.125] = 1.
    return [u]

def linear(nx=100, ny=100 ,Tmax=1., order=1, limiter='minmod'):

    dim = 2
# generate instance of class
    hcl = HyperbolicConsLawNumSolver(dim, order, limiter, True)

#set numerical Flux
    hcl.setNumericalFluxFuns(numFluxX_upwind, numFluxY_upwind, maxAbsEig)

# set boundary conditions
    boundaryCondFunN = "Neumann"
    boundaryCondFunS = "Neumann"
    boundaryCondFunW = "Neumann"
    boundaryCondFunE = "Neumann"
    hcl.setBoundaryCond(boundaryCondFunE, boundaryCondFunW, boundaryCondFunN, boundaryCondFunS)

# set initial state
    xCc = np.linspace(0.+.5/nx,1.-.5/nx,nx) # cell centers
    yCc = np.linspace(0.+.5/ny,1.-.5/ny,ny) # cell centers
    xv, yv = np.meshgrid(xCc, yCc)
    hcl.setUinit(initialCondFun(xv, yv), nx, ny, xCc, yCc)

# set flux parameters
    a_ = .5*np.ones((ny,nx+1))
    b_ = .5*np.ones((ny+1,nx))
    hcl.setFluxAndSourceParams(a = a_, b = b_)

    hcl.selfCheck()

# apply explicit time stepping
    t = 0.
# flux is linear, i.e., eigenvalues are independent of time
    eig = maxAbsEig(hcl)
    CFL = 0.49
    dt = 1.*CFL/eig
    while t<Tmax:
        if t+dt>Tmax:
            dt=Tmax-t
        t = hcl.timeStepExplicit(t, dt)

#plot result
    pl.title('linear 2d')
    pl.ion()
    pl.pcolor(xv, yv, hcl.getU(0), cmap='RdBu')

    return hcl

