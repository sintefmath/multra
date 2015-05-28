import pylab as pl
from GenericFVUtils import *


def maxAbsEig(hcl):
    return max( np.max(abs(hcl.params.u1))/hcl.dx, np.max(abs(hcl.params.u2))/hcl.dx  )

def numFluxX(self, U, dt, dx):

    Dxp_B1 =     U[0].u[U[0].order:-U[0].order,2:  ] - U[0].u[U[0].order:-U[0].order,1:-1]
    Dxm_B1 =     U[0].u[U[0].order:-U[0].order,1:-1] - U[0].u[U[0].order:-U[0].order, :-2]

    Dxp_B2 =     U[1].u[U[0].order:-U[0].order,2:  ] - U[1].u[U[0].order:-U[0].order,1:-1]
    Dxm_B2 =     U[1].u[U[0].order:-U[0].order,1:-1] - U[1].u[U[0].order:-U[0].order, :-2]

    u1p = np.maximum(self.params.u1[U[0].order:-U[0].order,U[0].order:-U[0].order], 0)
    u1m = np.minimum(self.params.u1[U[0].order:-U[0].order,U[0].order:-U[0].order], 0)

    F0 = u1m * Dxp_B1 + u1p * Dxm_B1
    F1 = u1m * Dxp_B2 + u1p * Dxm_B2

    return [F0, F1]

def numFluxY(self, U, dt, dy):

    Dyp_B1 =     U[0].u[2:  ,U[0].order:-U[0].order] - U[0].u[1:-1,U[0].order:-U[0].order]
    Dym_B1 =     U[0].u[1:-1,U[0].order:-U[0].order] - U[0].u[ :-2,U[0].order:-U[0].order]
                                                                             
    Dyp_B2 =     U[1].u[2:  ,U[0].order:-U[0].order] - U[1].u[1:-1,U[0].order:-U[0].order]
    Dym_B2 =     U[1].u[1:-1,U[0].order:-U[0].order] - U[1].u[ :-2,U[0].order:-U[0].order]

    u2p = np.maximum(self.params.u2[U[0].order:-U[0].order,U[0].order:-U[0].order], 0)
    u2m = np.minimum(self.params.u2[U[0].order:-U[0].order,U[0].order:-U[0].order], 0)

    F0 = u2m * Dyp_B1 + u2p * Dym_B1
    F1 = u2m * Dyp_B2 + u2p * Dym_B2

    return [F0, F1]

def numSource(self, U, dt, dx, dy):

    Dx0_u1 = .5/dx*(self.params.u1[U[0].order:-U[0].order,2:  ] - self.params.u1[U[0].order:-U[0].order, :-2])
    Dx0_u2 = .5/dx*(self.params.u2[U[0].order:-U[0].order,2:  ] - self.params.u2[U[0].order:-U[0].order, :-2])

    Dy0_u1 = .5/dy*(self.params.u1[2:  ,U[0].order:-U[0].order] - self.params.u1[ :-2,U[0].order:-U[0].order])
    Dy0_u2 = .5/dy*(self.params.u2[2:  ,U[0].order:-U[0].order] - self.params.u2[ :-2,U[0].order:-U[0].order])

    S0 = Dy0_u2 * U[0].u[U[0].order:-U[0].order,U[0].order:-U[0].order] - Dy0_u1 * U[1].u[U[1].order:-U[1].order,U[1].order:-U[1].order]
    S1 = Dx0_u2 * U[0].u[U[0].order:-U[0].order,U[0].order:-U[0].order] - Dx0_u1 * U[1].u[U[1].order:-U[1].order,U[1].order:-U[1].order]

    return [S0, S1]

def initialCondFun(xv, yv):
    uinit = np.zeros_like(xv)
    uinit[np.sqrt((xv-.25)**2 + (yv-.25)**2)<.125] = 1.
    u0_init = 1.*uinit
    u1_init = 1.*uinit
    return [u0_init, u1_init]

def linear(nx=100, ny=100, Tmax=1.):

    order = 1 
    limiter = None
    dim = 2
# generate instance of class
    hcl = HyperbolicConsLawNumSolver(dim, order, limiter, True, True)
    #hcl = HyperbolicConsLawNumSolver(dim, order, limiter, False, True)

    hcl.setNumericalFluxFuns(numFluxX, numFluxY, maxAbsEig)
    hcl.setNumericalSourceFun(numSource)

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
    #with boundary conditions
    u1_ = np.zeros((nx+2, ny+2))
    u2_ = np.zeros((nx+2, ny+2))
    u1_[1:-1,1:-1] = np.sin(np.pi*xv)
    u2_[1:-1,1:-1] = np.cos(np.pi*yv)

    hcl.setFluxAndSourceParams(u1 = u1_, u2 = u2_)

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
    pl.title('linear acoustics 2d')
    pl.ion()
    pl.pcolor(xv, yv, hcl.getU(0), cmap='RdBu')

    return hcl
