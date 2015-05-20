import pylab as pl
from GenericFVUtils import *


def maxAbsEig(hcl):
    return 3.*max( np.max(abs(hcl.params.u1))/hcl.dx, np.max(abs(hcl.params.u2))/hcl.dx  )

def numFluxX(self, U, dt, dx):

    Dxp_B1 =     U[0].u[U.order:-U.order,2:  ] - U[0].u[U.order:-U.order,1:-1]
    Dxm_B1 =     U[0].u[U.order:-U.order,1:-1] - U[0].u[U.order:-U.order, :-2]
    #Dx0_B1 = .5*(U[0].u[U.order:-U.order,2:  ] - U[0].u[U.order:-U.order, :-2])
    Dx0_u1 = .5*(self.params.u1[U.order:-U.order,2:  ] - self.params.u1[U.order:-U.order, :-2])
    Dx0_u2 = .5*(self.params.u2[U.order:-U.order,2:  ] - self.params.u2[U.order:-U.order, :-2])

    Dxp_B2 =     U[1].u[U.order:-U.order,2:  ] - U[1].u[U.order:-U.order,1:-1]
    Dxm_B2 =     U[1].u[U.order:-U.order,1:-1] - U[1].u[U.order:-U.order, :-2]
    #Dx0_B2 = .5*(U[1].u[U.order:-U.order,2:  ] - U[1].u[U.order:-U.order, :-2])

    Dyp_B1 =     U[0].u[2:  ,U.order:-U.order] - U[0].u[1:-1,U.order:-U.order]
    Dym_B1 =     U[0].u[1:-1,U.order:-U.order] - U[0].u[ :-2,U.order:-U.order]
    #Dy0_B1 = .5*(U[0].u[2:  ,U.order:-U.order] - U[0].u[ :-2,U.order:-U.order])
    Dy0_u1 = .5*(self.params.u1[2:  ,U.order:-U.order] - self.params.u1[ :-2,U.order:-U.order])
    Dy0_u2 = .5*(self.params.u2[2:  ,U.order:-U.order] - self.params.u2[ :-2,U.order:-U.order])
                                                                             
    Dyp_B2 =     U[1].u[2:  ,U.order:-U.order] - U[1].u[1:-1,U.order:-U.order]
    Dym_B2 =     U[1].u[1:-1,U.order:-U.order] - U[1].u[ :-2,U.order:-U.order]
    #Dy0_B2 = .5*(U[1].u[2:  ,U.order:-U.order] - U[1].u[ :-2,U.order:-U.order])


    u1p = np.maximum(self.params.u1,0)
    u1m = np.mainmum(self.params.u1,0)

    u2p = np.maximum(self.params.u2,0)
    u2m = np.mainmum(self.params.u2,0)

    F0 = - u1m * Dxp_B1 - u1p * Dxm_B1 - u2m * Dyp_B1 - u2p * Dym_B1 \
            -Dy0_u2 * B1 + Dy0_u1 * B2
    F1 = - u1m * Dxp_B2 - u1p * Dxm_B2 - u2m * Dyp_B2 - u2p * Dym_B2 \
            -Dx0_u2 * B1 + Dx0_u1 * B2

    return [F0, F1]

def numFluxy(self, U, dt, dx):
    tmp = np.zeros_like(U
    return 

def boundaryCondFunW(t, dx, y):
# square pulse
    u = 0.*y
    if t<1.0:
        #u = 0.25
        u[(y>.45) & (y<.55)] = np.sin(2*2*np.pi*t)
    return [0.*y, u]

def initialCondFun(xv, yv):
    #uinit = .1*np.ones((ny, nx))
    #uinit[np.sqrt((xv-.25)**2 + (yv-.25)**2)<.125] = 1.
    u0_init = np.zeros_like(xv)
    u1_init = np.zeros_like(xv)
    return [u0_init, u1_init]

def linear(nx=100, ny=100 ,Tmax=1.):

    order = 1 
    limiter = None
# generate instance of class
    hcl = HyperbolicConsLaw(order, limiter, True)

    hcl.setNumericalFluxFuns(numFluxX, None, maxAbsEig)

# set boundary conditions
    boundaryCondFunN = "Neumann"
    boundaryCondFunS = "Neumann"
    #boundaryCondFunW = "Neumann"
    boundaryCondFunE = "Neumann"
    hcl.setBoundaryCond(boundaryCondFunE, boundaryCondFunW, boundaryCondFunN, boundaryCondFunS)

# set initial state
    xCc = np.linspace(0.+.5/nx,1.-.5/nx,nx) # cell centers
    yCc = np.linspace(0.+.5/ny,1.-.5/ny,ny) # cell centers
    xv, yv = np.meshgrid(xCc, yCc)
    hcl.setUinit(initialCondFun(xv, yv), nx, ny, xCc, yCc)

# set flux parameters
    xCi = np.linspace(1.+.5,1.+-.5,nx+1) # cell interface
    yCi = np.linspace(1.+.5,1.+-.5,ny) # cell interface
    xvi, yvi = np.meshgrid(xCi, yCi)
    u0_ = 0*xvi

    #u0_ = np.ones((ny, nx+1))
    #u0_[xvi<0] = -1

    xCi = np.linspace(1.+.5,1.+-.5,nx) # cell interface
    yCi = np.linspace(1.+.5,1.+-.5,ny+1) # cell interface
    xvi, yvi = np.meshgrid(xCi, yCi)
    v0_ = 0*yvi

    #v0_ = np.ones((ny+1, nx))
    #v0_[yvi<0] = -1

    hcl.setFluxParams(u1 = u1_, u2 = u2_)

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
