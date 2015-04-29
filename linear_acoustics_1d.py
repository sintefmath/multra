import pylab as pl
from GenericFVUtils import *

def maxAbsEig(self, U, dx, dy):
    return np.max(np.abs(np.max(self.params.V) - self.params.c0), np.abs(np.min(self.params.V) + self.params.c0))/dx

#def FluxX(u):
#    return [V*u[0] + K0*u[1], 1./rho0*u[0] + V*u[1]]

def numFluxX_LxF(self, U, dt, dx):
    F0 = 0.5*( (      self.params.V*U[0].uW + self.params.K0*U[1].uW) + (      self.params.V*U[0].uE + self.params.K0*U[1].uE) ) - 0.5*dx/dt*(U[0].uE - U[0].uW)
    F1 = 0.5*( (1./self.params.rho0*U[0].uW +  self.params.V*U[1].uW) + (1./self.params.rho0*U[0].uE +  self.params.V*U[1].uE) ) - 0.5*dx/dt*(U[1].uE - U[1].uW)
    return [F0, F1]

def numFluxX_upwind(self, U, dt, dx):

    sL = self.params.V - self.params.c0
    sR = self.params.V + self.params.c0

    mask_plus = sL > 0
    mask_minus = sR < 0
    mask_middle = -mask_plus & -mask_minus

    F0W  = self.params.V*U[0].uW + self.params.K0*U[1].uW
    F1W = 1./self.params.rho0*U[0].uW + self.params.V*U[1].uW

    F0E = self.params.V*U[0].uE + self.params.K0*U[1].uE
    F1E = 1./self.params.rho0*U[0].uE + self.params.V*U[1].uE

    F0 = np.empty_like(U[0].uW)
    F1 = np.empty_like(U[1].uW)

    [F0[mask_plus], F1[mask_plus]] = [F0W[mask_plus], F1W[mask_plus]]
    [F0[mask_minus], F1[mask_minus]] = [F0W[mask_minus], F1W[mask_minus]]
    [F0[mask_middle], F1[mask_middle]] = [
        (sR*F0W[mask_middle] - sL*F0E[mask_middle] + sL*sR*(U[0].uE[mask_middle] - U[0].uW[mask_middle]) )/(2.*self.params.c0),
        (sR*F1W[mask_middle] - sL*F1E[mask_middle] + sL*sR*(U[1].uE[mask_middle] - U[1].uW[mask_middle]) )/(2.*self.params.c0)]

    return [F0, F1]

def boundaryCondFunE(t, dx, y):
    return [0., 0.]
def boundaryCondFunW(t, dx, y):
# square pulse
    u = 0.
    if (t<0.4):
        u = 0.25
    return [0., u]

def linear(nx=1000, Tmax=1., order=1, limiter='minmod', method='upwind'):

# generate instance of class
    hcl = HyperbolicConsLaw(order, limiter)

#set numerical Flux
    if method=='LxF':
        numFluxX = numFluxX_LxF
    else:
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
    u0_init = np.zeros((nx))
    u1_init = np.zeros((nx))
    #u0_init[order:-order] = np.sin(2*np.pi*xCc)
    hcl.setU([u0_init, u1_init], nx, ny, xCc, yCc)


# set flux parameters
    xCi = np.linspace(0,1,nx+1) # cell interfaces
    rho0_ = 10.
    K0_ = 1.
    c0_ = np.sqrt(K0_/rho0_)
    #V_ = -xCi + .5
    V_ = -xCi*0.
    hcl.setFluxParams(rho0 = rho0_, K0 = K0_, c0 = c0_, V = V_)

# apply explicit time stepping
    t = 0.
    while t<Tmax:
        t = hcl.timeStepExplicit(t, Tmax)

#plot result
    pl.figure(2)
    pl.title('linear acoustics 1d')
    pl.ion()
    pl.plot(hcl.xCc,hcl.getU(0))
    pl.plot(xCi,V_,'k:')

    return hcl

