import pylab as pl
from GenericFVUtils import *

rho0_ = 10.
K0_ = 1.
c0_ = np.sqrt(K0_/rho0_)

def maxAbsEig(hcl):
    return max( np.max(abs(hcl.params.u0 - hcl.params.c0)), np.max(abs(hcl.params.u0 + hcl.params.c0)) )/hcl.dx
#for time dependent flux use
#    def maxAbsEig(self, U, dx, dy):
#        return max( np.max(abs(self.params.u0 - self.params.c0)), np.max(abs(self.params.u0 + self.params.c0)) )/dx


#def FluxX(u):
#    return [u0*u[0] + K0*u[1], 1./rho0*u[0] + u0*u[1]]

def numFluxX_LxF(self, U, dt, dx):
    F0 = 0.5*( (     self.params.u0*U[0].uW + self.params.K0*U[1].uW) + (     self.params.u0*U[0].uE + self.params.K0*U[1].uE) ) - 0.5*dx/dt*(U[0].uE - U[0].uW)
    F1 = 0.5*( (1./self.params.rho0*U[0].uW + self.params.u0*U[1].uW) + (1./self.params.rho0*U[0].uE + self.params.u0*U[1].uE) ) - 0.5*dx/dt*(U[1].uE - U[1].uW)
    return [F0, F1]

def numFluxX_upwind(self, U, dt, dx):

    sL = self.params.u0 - self.params.c0
    sR = self.params.u0 + self.params.c0

    mask_plus = sL > 0
    mask_minus = sR < 0
    mask_middle = -mask_plus & -mask_minus

    F0W = self.params.u0*U[0].uW + self.params.K0*U[1].uW
    F1W = 1./self.params.rho0*U[0].uW + self.params.u0*U[1].uW

    F0E = self.params.u0*U[0].uE + self.params.K0*U[1].uE
    F1E = 1./self.params.rho0*U[0].uE + self.params.u0*U[1].uE

    F0 = np.empty_like(U[0].uW)
    F1 = np.empty_like(U[1].uW)

    [F0[mask_plus], F1[mask_plus]] = [F0W[mask_plus], F1W[mask_plus]]
    [F0[mask_minus], F1[mask_minus]] = [F0E[mask_minus], F1E[mask_minus]]
    [F0[mask_middle], F1[mask_middle]] = [
        (sR[mask_middle]*F0W[mask_middle] - sL[mask_middle]*F0E[mask_middle] + sL[mask_middle]*sR[mask_middle]*(U[0].uE[mask_middle] - U[0].uW[mask_middle]) )/(2.*self.params.c0),
        (sR[mask_middle]*F1W[mask_middle] - sL[mask_middle]*F1E[mask_middle] + sL[mask_middle]*sR[mask_middle]*(U[1].uE[mask_middle] - U[1].uW[mask_middle]) )/(2.*self.params.c0)]

    return [F0, F1]

def boundaryCondFunW(t, dx, y):
# square pulse
    u = 0.
    if (t<1.0):
        #u = 0.25
        u = np.sin(2*2*np.pi*t)
    return [0, u]

def initialCondFun(x):
    #u0_init[order:-order] = np.sin(2*np.pi*xCc)
    #u0_init[xCc<.5] = 1.
    u0_init = np.zeros_like(x)
    u1_init = np.zeros_like(x)
    return [u0_init, u1_init]

def linear(nx=1000, Tmax=1., order=1, limiter='minmod', method='upwind'):

# generate instance of class
    hcl = HyperbolicConsLaw(order, limiter, True)

#set numerical Flux
    if method=='LxF':
        numFluxX = numFluxX_LxF
    else:
        numFluxX = numFluxX_upwind
    numFluxY = None
    hcl.setNumericalFluxFuns(numFluxX, numFluxY, maxAbsEig)

# set boundary conditions
    boundaryCondFunE = "Neumann"
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
    #u0_ = np.linspace(1,-1,nx+1)
    u0_ = -xCi + .5
    u0_ *= .0
    hcl.setFluxParams(rho0 = rho0_, K0 = K0_, c0 = c0_, u0 = u0_)

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
    pl.title('linear acoustics 1d')
    pl.ion()
    pl.plot(hcl.xCc,hcl.getU(0))
    pl.plot(xCi,u0_,'k:')

    return hcl

