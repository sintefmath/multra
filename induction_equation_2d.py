import pylab as pl
from GenericFVUtils import *


def maxAbsEig(hcl):
    return max( np.max(abs(hcl.params.u1))/hcl.dx, np.max(abs(hcl.params.u2))/hcl.dy  )

def numFluxX(self, U, dt, dx):
    
    od = U[0].order

    Dx0_B1 = (U[0].u[od:-od,2: ] - U[0].u[od:-od, :-2])/2.
    Dx0_B2 = (U[1].u[od:-od,2: ] - U[1].u[od:-od, :-2])/2.

    Dx0_u1 = (self.params.u1[od:-od,2: ] - self.params.u1[od:-od, :-2])/2.
    Dx0_u2 = (self.params.u2[od:-od,2: ] - self.params.u2[od:-od, :-2])/2.

    Dxx_B1 = (U[0].u[od:-od,2: ] - 2.*U[0].u[od:-od,od:-od] + U[0].u[od:-od, :-2])/2.
    Dxx_B2 = (U[1].u[od:-od,2: ] - 2.*U[1].u[od:-od,od:-od] + U[1].u[od:-od, :-2])/2.

    maxAbs_u1 = np.maximum(np.abs(self.params.u1[od:-od,od:-od]), dx/2.)

    F0 = - self.params.u1[od:-od,od:-od] * Dx0_B1 \
         + maxAbs_u1 * Dxx_B1
    F1 = - self.params.u1[od:-od,od:-od] * Dx0_B2 \
         + Dx0_u2 * U[0].u[od:-od,od:-od] \
         - Dx0_u1 * U[1].u[od:-od,od:-od] \
         + maxAbs_u1 * Dxx_B2

    return [-F0, -F1]

def numFluxY(self, U, dt, dy):

    
    od = U[0].order

    Dy0_B1 = (U[0].u[2:, od:-od] - U[0].u[:-2, od:-od])/2.
    Dy0_B2 = (U[1].u[2:, od:-od] - U[1].u[:-2, od:-od])/2.

    Dy0_u1 = (self.params.u1[2:, od:-od] - self.params.u1[:-2, od:-od])/2.
    Dy0_u2 = (self.params.u2[2:, od:-od] - self.params.u2[:-2, od:-od])/2.

    Dyy_B1 = (U[0].u[2:, od:-od] - 2.*U[0].u[od:-od,od:-od] + U[0].u[:-2, od:-od])/2.
    Dyy_B2 = (U[1].u[2:, od:-od] - 2.*U[1].u[od:-od,od:-od] + U[1].u[:-2, od:-od])/2.

    maxAbs_u2 = np.maximum(np.abs(self.params.u2[od:-od,od:-od]), dy/2.)

    F0 = - self.params.u2[od:-od,od:-od] * Dy0_B1 \
         - Dy0_u2 * U[0].u[od:-od,od:-od] \
         + Dy0_u1 * U[1].u[od:-od,od:-od] \
         + maxAbs_u2 * Dyy_B1
    F1 = - self.params.u2[od:-od,od:-od] * Dy0_B2 \
         + maxAbs_u2 * Dyy_B2 

    return [-F0, -F1]

def numFluxX_form2(self, U, dt, dx):
    
    od = U[0].order

    Dxp_B1 = U[0].u[od:-od,2:  ] - U[0].u[od:-od,1:-1]
    Dxm_B1 = U[0].u[od:-od,1:-1] - U[0].u[od:-od, :-2]

    Dxp_B2 = U[1].u[od:-od,2:  ] - U[1].u[od:-od,1:-1]
    Dxm_B2 = U[1].u[od:-od,1:-1] - U[1].u[od:-od, :-2]

    u1p = np.maximum(self.params.u1[od:-od,od:-od], 0)
    u1m = np.minimum(self.params.u1[od:-od,od:-od], 0)

    F0 = u1m * Dxp_B1 + u1p * Dxm_B1
    F1 = u1m * Dxp_B2 + u1p * Dxm_B2

    return [F0, F1]

def numFluxY_form2(self, U, dt, dy):

    od = U[0].order

    Dyp_B1 = U[0].u[2:  ,od:-od] - U[0].u[1:-1,od:-od]
    Dym_B1 = U[0].u[1:-1,od:-od] - U[0].u[ :-2,od:-od]
                                                                         
    Dyp_B2 = U[1].u[2:  ,od:-od] - U[1].u[1:-1,od:-od]
    Dym_B2 = U[1].u[1:-1,od:-od] - U[1].u[ :-2,od:-od]

    u2p = np.maximum(self.params.u2[od:-od,od:-od], 0)
    u2m = np.minimum(self.params.u2[od:-od,od:-od], 0)

    F0 = u2m * Dyp_B1 + u2p * Dym_B1
    F1 = u2m * Dyp_B2 + u2p * Dym_B2

    return [F0, F1]

def numSource_form2(self, U, dt, dx, dy):

    od = U[0].order

    Dx0_u1 = (self.params.u1[od:-od,2:  ] - self.params.u1[od:-od, :-2])/(2.*dx)
    Dx0_u2 = (self.params.u2[od:-od,2:  ] - self.params.u2[od:-od, :-2])/(2.*dx)

    Dy0_u1 = (self.params.u1[2:  ,od:-od] - self.params.u1[ :-2,od:-od])/(2.*dy)
    Dy0_u2 = (self.params.u2[2:  ,od:-od] - self.params.u2[ :-2,od:-od])/(2.*dy)

    S0 = - Dy0_u2 * U[0].u[od:-od,od:-od] + Dy0_u1 * U[1].u[od:-od,od:-od]
    S1 = + Dx0_u2 * U[0].u[od:-od,od:-od] - Dx0_u1 * U[1].u[od:-od,od:-od]

    return [S0, S1]

def initialCondFun_linear(xv, yv):
    uinit = np.zeros_like(xv)
    uinit[xv>yv] = 2.
    return [uinit, uinit]
def initialCondParamsFun_linear(xv, yv, dim, order, boundaryCondFunN, boundaryCondFunS, boundaryCondFunW, boundaryCondFunE):
    [ny, nx] = xv.shape

    u1_ = 1. + np.zeros((ny+2, nx+2))
    u1_ = apply_BC_W(u1_, boundaryCondFunW, dim, order)
    u1_ = apply_BC_E(u1_, boundaryCondFunE, dim, order)
    u1_ = apply_BC_N(u1_, boundaryCondFunN, order)
    u1_ = apply_BC_S(u1_, boundaryCondFunS, order)

    u2_ = 2. + np.zeros((ny+2, nx+2))
    u2_ = apply_BC_W(u2_, boundaryCondFunW, dim, order)
    u2_ = apply_BC_E(u2_, boundaryCondFunE, dim, order)
    u2_ = apply_BC_N(u2_, boundaryCondFunN, order)
    u2_ = apply_BC_S(u2_, boundaryCondFunS, order)
    return [u1_, u2_]

def initialCondFun_potField(xv, yv):
    uinit = np.zeros_like(xv)
    return [1. + .25*(np.cos(2.*np.pi*xv) + 2.*np.sin(2.*np.pi*yv)), np.sin(2.*np.pi*xv) + 2.*np.cos(2.*np.pi*yv)]
def initialCondParamsFun_potField(xv, yv, dim, order, boundaryCondFunN, boundaryCondFunS, boundaryCondFunW, boundaryCondFunE):
    [ny, nx] = xv.shape
    xCc_gc = np.linspace(-.5+.5/nx-1./nx*order, .5-.5/nx+1./nx*order, nx+2*order) # cell centers with ghost cells
    yCc_gc = np.linspace(-.5+.5/ny-1./ny*order, .5-.5/ny+1./ny*order, ny+2*order) # cell centers
    xv_gc, yv_gc = np.meshgrid(xCc_gc, yCc_gc)

    u1_ = 1. + np.sin(2.*np.pi*xv_gc)*np.cos(2.*np.pi*yv_gc)
    u1_ = apply_BC_W(u1_, boundaryCondFunW, dim, order)
    u1_ = apply_BC_E(u1_, boundaryCondFunE, dim, order)
    u1_ = apply_BC_N(u1_, boundaryCondFunN, order)
    u1_ = apply_BC_S(u1_, boundaryCondFunS, order)

    u2_ = 1. - np.cos(2.*np.pi*xv_gc)*np.sin(2.*np.pi*yv_gc)
    u2_ = apply_BC_W(u2_, boundaryCondFunW, dim, order)
    u2_ = apply_BC_E(u2_, boundaryCondFunE, dim, order)
    u2_ = apply_BC_N(u2_, boundaryCondFunN, order)
    u2_ = apply_BC_S(u2_, boundaryCondFunS, order)
    return [u1_, u2_]

def initialCondFun_rot(xv, yv):
    uinit = np.zeros_like(xv)
    ts_xv = 2.*(xv-.5)
    ts_yv = 2.*(yv-.5)
    factor = 4*np.exp(-20.*( (ts_xv - .5)**2 + ts_yv**2 ) )
    return [-factor*ts_yv, factor*(ts_xv - .5)]
def initialCondParamsFun_rot(xv, yv, dim, order, boundaryCondFunN, boundaryCondFunS, boundaryCondFunW, boundaryCondFunE):
    [ny, nx] = xv.shape
    xCc_gc = np.linspace(-.5+.5/nx-1./nx*order, .5-.5/nx+1./nx*order, nx+2*order) # cell centers with ghost cells
    yCc_gc = np.linspace(-.5+.5/ny-1./ny*order, .5-.5/ny+1./ny*order, ny+2*order) # cell centers
    xv_gc, yv_gc = np.meshgrid(xCc_gc, yCc_gc)
    return [-yv_gc, xv_gc]

def initialCondFun_OT(xv, yv):
    u0_init = -np.sin(2.*np.pi*yv)
    u1_init = +np.sin(4.*np.pi*xv)
    return [u0_init, u1_init]
def initialCondParamsFun_OT(xv, yv, dim, order, boundaryCondFunN, boundaryCondFunS, boundaryCondFunW, boundaryCondFunE):
    [ny, nx] = xv.shape
    u1_ = np.zeros((ny+2, nx+2))
    u1_[1:-1,1:-1] = -np.sin(2.*np.pi*yv)
    u1_ = apply_BC_W(u1_, boundaryCondFunW, dim, order)
    u1_ = apply_BC_E(u1_, boundaryCondFunE, dim, order)
    u1_ = apply_BC_N(u1_, boundaryCondFunN, order)
    u1_ = apply_BC_S(u1_, boundaryCondFunS, order)

    u2_ = np.zeros((ny+2, nx+2))
    u2_[1:-1,1:-1] = +np.sin(2.*np.pi*xv)
    u2_ = apply_BC_W(u2_, boundaryCondFunW, dim, order)
    u2_ = apply_BC_E(u2_, boundaryCondFunE, dim, order)
    u2_ = apply_BC_N(u2_, boundaryCondFunN, order)
    u2_ = apply_BC_S(u2_, boundaryCondFunS, order)

    return [u1_, u2_]

def linear(nx=100, ny=100, Tmax=1.,example=1):

    order = 1 
    limiter = None
    dim = 2
# generate instance of class
    hcl = HyperbolicConsLawNumSolver(dim, order, limiter, True, True)

    hcl.setNumericalFluxFuns(numFluxX, numFluxY, maxAbsEig)
    #hcl.setNumericalSourceFun(numSource)

# set boundary conditions
    xCc = np.linspace(0.+.5/nx,1.-.5/nx,nx) # cell centers
    yCc = np.linspace(0.+.5/ny,1.-.5/ny,ny) # cell centers
    xv, yv = np.meshgrid(xCc, yCc)

    if example == 1:
        print("case: linear advection")
        boundaryCondFunE = "Neumann"
        boundaryCondFunW = "Neumann"
        boundaryCondFunN = "Neumann"
        boundaryCondFunS = "Neumann"
        initialCondFun = initialCondFun_linear
        initialCondParamsFun = initialCondParamsFun_linear
    elif example == 2:
        print("case: potential magentic field")
        boundaryCondFunN = "periodic"
        boundaryCondFunS = "periodic"
        boundaryCondFunW = "periodic"
        boundaryCondFunE = "periodic"
        initialCondFun = initialCondFun_potField
        initialCondParamsFun = initialCondParamsFun_potField
    elif example == 3:
        print("case: rotation around origin")
        boundaryCondFunN = "periodic"
        boundaryCondFunS = "periodic"
        boundaryCondFunW = "periodic"
        boundaryCondFunE = "periodic"
        initialCondFun = initialCondFun_rot
        initialCondParamsFun = initialCondParamsFun_rot
    elif example == 4:
        print("case: Orszag-Tang")
        boundaryCondFunN = "periodic"
        boundaryCondFunS = "periodic"
        boundaryCondFunW = "periodic"
        boundaryCondFunE = "periodic"
        initialCondFun = initialCondFun_OT
        initialCondParamsFun = initialCondParamsFun_OT

    hcl.setUinit(initialCondFun(xv, yv), nx, ny, xCc, yCc)
    [u1_, u2_] = initialCondParamsFun(xv, yv, dim, order, boundaryCondFunN, boundaryCondFunS, boundaryCondFunW, boundaryCondFunE)

# set initial state

    hcl.setBoundaryCond(boundaryCondFunE, boundaryCondFunW, boundaryCondFunN, boundaryCondFunS)
    hcl.setFluxAndSourceParams(u1 = u1_, u2 = u2_)

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
    pl.title('induction equation 2d')
    pl.ion()
    #pl.figure(1)
    #pl.pcolor(xv, yv, hcl.getU(0), cmap='RdBu')
    #pl.colorbar()
    #pl.figure(2)
    pl.pcolor(xv, yv, hcl.getU(1), cmap='RdBu')
    #pl.colorbar()

    [ca, cb] = initialCondFun(xv, yv)
    print("abs cons. error = ", abs(hcl.dx*hcl.dy*(np.sum(ca) - np.sum(hcl.getU(0)))) , abs(hcl.dx*hcl.dy*(np.sum(cb) - np.sum(hcl.getU(1)))) )

    print("rel cons. error = ", abs(np.sum(ca) - np.sum(hcl.getU(0)))/abs(np.sum(ca)) , abs(np.sum(cb) - np.sum(hcl.getU(1)))/abs(np.sum(cb)) )


    return hcl
