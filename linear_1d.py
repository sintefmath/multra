import pylab as pl
from GenericFVUtils import *

def maxAbsEig(hcl):
    return np.max(np.abs(hcl.params.a))/hcl.dx

def FluxX(self, u, m):
    return self.params.a[m]*u[m]

def numFluxX_upwind(self, U, dt, dx):
    mask = self.params.a < 0.
    Fu = np.empty_like(U[0].uW)
    Fu[-mask] = self.params.a[-mask]*U[0].uW[-mask]
    Fu[ mask] = self.params.a[ mask]*U[0].uE[ mask]
    return [Fu]

#def boundaryCondFunE(t, dx, y):
#    return [0.]
#
#def boundaryCondFunW(t, dx, y):
## square pulse
#    u = 0.
#    if (t<0.4):
#        u = 0.25
#    return [u]

def initialCondFun(x):
    u = np.zeros_like(x)
    u[np.abs(x-.25)<.125] = 1.
    return [u]


def linear(nx=1000, Tmax=1., order=1, limiter='minmod'):

# generate instance of class
    hcl = HyperbolicConsLaw(order, limiter, True)

#set numerical Flux
    numFluxX = numFluxX_upwind
    numFluxY = None
    hcl.setNumericalFluxFuns(numFluxX, numFluxY, maxAbsEig)

# set boundary conditions
    boundaryCondFunE = "Neumann"
    boundaryCondFunW = "Neumann"
    boundaryCondFunN = "Neumann"
    boundaryCondFunS = "Neumann"
    hcl.setBoundaryCond(boundaryCondFunE, boundaryCondFunW, boundaryCondFunN, boundaryCondFunS)

# set initial state
    xCc = np.linspace(0.+.5/nx,1.-.5/nx,nx) # cell centers
    yCc = None
    ny = None
    hcl.setUinit(initialCondFun(xCc), nx, ny, xCc, yCc)

# set flux parameters
    xCi = np.linspace(0,1,nx+1) # cell interfaces
    #a_ = -xCi + .5
    a_ = .25*np.ones_like(xCi)
    hcl.setFluxParams(a = a_)

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
    pl.title('linear 1d')
    pl.ion()
    pl.plot(hcl.xCc,hcl.getU(0))
    pl.plot(xCi,a_,'k:')

    return hcl
