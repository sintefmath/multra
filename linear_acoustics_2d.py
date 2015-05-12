import pylab as pl
from GenericFVUtils import *

def maxAbsEig(hcl):
    return max( \
                max( np.max(abs(hcl.params.u0 - hcl.params.c0)), np.max(abs(hcl.params.u0 + hcl.params.c0)) )/hcl.dx \
                , \
                max( np.max(abs(hcl.params.v0 - hcl.params.c0)), np.max(abs(hcl.params.v0 + hcl.params.c0)) )/hcl.dy \
           )

def numFluxX_HLL2(self, U, dt, dx):

    sL = self.params.u0 - self.params.c0
    sR = self.params.u0 + self.params.c0

    mask_plus = sL > 0
    mask_minus = sR < 0
    mask_middle = -mask_plus & -mask_minus
    #mask_middleL = middle & self.params.u0<=0
    #mask_middleR = middle & self.params.u0>0

    F0W = self.params.u0*U[0].uW + self.params.K0*U[1].uW
    F1W = 1./self.params.rho0*U[0].uW + self.params.u0*U[1].uW
    F2W = 0.*self.params.u0*U[2].uW

    F0E = self.params.u0*U[0].uE + self.params.K0*U[1].uE
    F1E = 1./self.params.rho0*U[0].uE + self.params.u0*U[1].uE
    F2E = self.params.u0*U[2].uE

    F0 = np.empty_like(U[0].uW)
    F1 = np.empty_like(U[1].uW)
    F2 = np.empty_like(U[2].uW)

    [F0[mask_plus], F1[mask_plus], F2[mask_plus]] = [F0W[mask_plus], F1W[mask_plus], F2W[mask_plus]]
    [F0[mask_minus], F1[mask_minus], F2[mask_minus]] = [F0E[mask_minus], F1E[mask_minus], F2E[mask_minus]]
    [F0[mask_middle], F1[mask_middle], F2[mask_middle]] = [
        (sR[mask_middle]*F0W[mask_middle] - sL[mask_middle]*F0E[mask_middle] + sL[mask_middle]*sR[mask_middle]*(U[0].uE[mask_middle] - U[0].uW[mask_middle]) )/(2.*self.params.c0),
        (sR[mask_middle]*F1W[mask_middle] - sL[mask_middle]*F1E[mask_middle] + sL[mask_middle]*sR[mask_middle]*(U[1].uE[mask_middle] - U[1].uW[mask_middle]) )/(2.*self.params.c0),
        (sR[mask_middle]*F2W[mask_middle] - sL[mask_middle]*F2E[mask_middle] + sL[mask_middle]*sR[mask_middle]*(U[2].uE[mask_middle] - U[2].uW[mask_middle]) )/(2.*self.params.c0)
        ]

    return [F0, F1, F2]


def numFluxY_HLL2(self, U, dt, dy):

    sL = self.params.v0 - self.params.c0
    sR = self.params.v0 + self.params.c0

    mask_plus = sL > 0
    mask_minus = sR < 0
    mask_middle = -mask_plus & -mask_minus
    #mask_middleL = middle & self.params.v0<=0
    #mask_middleR = middle & self.params.v0>0

    F0S  = self.params.v0*U[0].uS + self.params.K0*U[2].uS
    F1S = self.params.v0*U[1].uS
    F2S = 1./self.params.rho0*U[0].uS + self.params.v0*U[2].uS

    F0N  = self.params.v0*U[0].uN + self.params.K0*U[2].uN
    F1N = self.params.v0*U[1].uN
    F2N = 1./self.params.rho0*U[0].uN + self.params.v0*U[2].uN

    F0 = np.empty_like(U[0].uS)
    F1 = np.empty_like(U[1].uS)
    F2 = np.empty_like(U[2].uS)

    [F0[mask_plus], F1[mask_plus], F2[mask_plus]] = [F0S[mask_plus], F1S[mask_plus], F2S[mask_plus]]
    [F0[mask_minus], F1[mask_minus], F2[mask_minus]] = [F0S[mask_minus], F1S[mask_minus], F2S[mask_minus]]
    [F0[mask_middle], F1[mask_middle], F2[mask_middle]] = [
        (sR[mask_middle]*F0S[mask_middle] - sL[mask_middle]*F0N[mask_middle] + sL[mask_middle]*sR[mask_middle]*(U[0].uN[mask_middle] - U[0].uS[mask_middle]) )/(2.*self.params.c0),
        (sR[mask_middle]*F1S[mask_middle] - sL[mask_middle]*F1N[mask_middle] + sL[mask_middle]*sR[mask_middle]*(U[1].uN[mask_middle] - U[1].uS[mask_middle]) )/(2.*self.params.c0),
        (sR[mask_middle]*F2S[mask_middle] - sL[mask_middle]*F2N[mask_middle] + sL[mask_middle]*sR[mask_middle]*(U[2].uN[mask_middle] - U[2].uS[mask_middle]) )/(2.*self.params.c0)
        ]

    return [F0, F1, F2]

def boundaryCondFunW(t, dx, y):
# square pulse
    u = 0.*y
    if t<1.0:
        #u = 0.25
        u[(y>.45) & (y<.55)] = np.sin(2*2*np.pi*t)
    return [0.*y, u, 0.*y]

def linear(nx=100, ny=100 ,Tmax=1., order=1, limiter='minmod', method='HLL2'):

# generate instance of class
    hcl = HyperbolicConsLaw(order, limiter, True)

#set numerical Flux
    if method=='HLL2':
        numFluxX = numFluxX_HLL2
        numFluxY = numFluxY_HLL2
    else:
        numFluxX = numFluxX_HLL3
        numFluxY = numFluxY_HLL3

    hcl.setNumericalFluxFuns(numFluxX, numFluxY, maxAbsEig)

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
    uinit = .1*np.ones((ny, nx))
    uinit[np.sqrt((xv-.25)**2 + (yv-.25)**2)<.125] = 1.
    hcl.setUinit([0*uinit, 0.*uinit, 0.*uinit], nx, ny, xCc, yCc)

# set flux parameters
    rho0_ = 10.
    K0_ = 1.
    c0_ = np.sqrt(K0_/rho0_)

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

    hcl.setFluxParams(rho0 = rho0_, K0 = K0_, c0 = c0_, u0 = u0_, v0 = v0_)

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
