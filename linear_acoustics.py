import pylab as pl
from GenericFVUtils import *

rho0 = None
K0 = None
c0 = None
V = None

def maxEig(U, dx, dy):
    global c0
    global V
    return np.max(np.abs(np.max(V) - c0), np.abs(np.min(V) + c0))/dx

#def FluxX(u):
#    global rho0
#    global K0
#    global V
#    return [V*u[0] + K0*u[1], 1./rho0*u[0] + V*u[1]]

def numFluxX_LxF(U, dt, dx):
    global rho0
    global K0
    global V
    F0 = 0.5*( (      V*U[0].uW + K0*U[1].uW) + (      V*U[0].uE + K0*U[1].uE) ) - 0.5*dx/dt*(U[0].uE - U[0].uW)
    F1 = 0.5*( (1./rho0*U[0].uW +  V*U[1].uW) + (1./rho0*U[0].uE +  V*U[1].uE) ) - 0.5*dx/dt*(U[1].uE - U[1].uW)
    return [F0, F1]

def numFluxX_upwind(U, dt, dx):
    global rho0
    global K0
    global c0
    global V

    sL = V - c0
    sR = V + c0

    mask_plus = sL > 0
    mask_minus = sR < 0
    mask_middle = -mask_plus & -mask_minus

    F0W  = V*U[0].uW + K0*U[1].uW
    F1W = 1./rho0*U[0].uW + V*U[1].uW

    F0E = V*U[0].uE + K0*U[1].uE
    F1E = 1./rho0*U[0].uE + V*U[1].uE

    F0 = np.empty_like(U[0].uW)
    F1 = np.empty_like(U[1].uW)

    [F0[mask_plus], F1[mask_plus]] = [F0W[mask_plus], F1W[mask_plus]]
    [F0[mask_minus], F1[mask_minus]] = [F0W[mask_minus], F1W[mask_minus]]
    [F0[mask_middle], F1[mask_middle]] = [
        (sR*F0W[mask_middle] - sL*F0E[mask_middle] + sL*sR*(U[0].uE[mask_middle] - U[0].uW[mask_middle]) )/(2.*c0),
        (sR*F1W[mask_middle] - sL*F1E[mask_middle] + sL*sR*(U[1].uE[mask_middle] - U[1].uW[mask_middle]) )/(2.*c0)]

    return [F0, F1]

def boundaryCondFunE(t, dx, y):
    return [0., 0.]
def boundaryCondFunW(t, dx, y):
    u = 0.
    x0 = 0.4
    if (t<x0):
        u = 0.25
    return [0., u]

def linearProfiling(nx=1000, Tmax=1., order=1, limiter='minmod', method='upwind'):
    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput
    graphviz = GraphvizOutput()
    graphviz.output_file = 'basic.png'
    with PyCallGraph(output=graphviz):
        print("Starting simulation...")
        linear(nx, Tmax, order, limiter, method, False)
        print("... done.")


def linear(nx=1000, Tmax=1., order=1, limiter='minmod', method='upwind', plotResult = True):

    hcl = HyperbolicConsLaw()

    if method=='LxF':
        numFluxX = numFluxX_LxF
    else:
        numFluxX = numFluxX_upwind

    xCi = np.linspace(0,1,nx+1) # cell interfaces
    dx = xCi[1]-xCi[0]

    ny = None
    dy = None
    numFluxY = None
    boundaryCondFunN = None
    boundaryCondFunS = None
    #boundaryCondFunW = None
    #boundaryCondFunE = None

    hcl.setFuns(maxEig, numFluxX, numFluxY, boundaryCondFunE, boundaryCondFunW, boundaryCondFunN, boundaryCondFunS, order, limiter)

    xCc = np.linspace(0.+dx/2.,1.-dx/2.,nx) # cell centers
    yCc = np.array((0.,0.))

    u0_init = np.zeros((nx))
    u1_init = np.zeros((nx))
    #u0_init[order:-order] = np.sin(2*np.pi*xCc)

    hcl.setU([u0_init, u1_init], nx, ny, xCc, yCc)

    global rho0
    global K0
    global c0
    global V
    rho0 = 10.
    K0 = 1.
    c0 = np.sqrt(K0/rho0)
    #V = -xCi + .5
    V = -xCi*0.

    #pl.plot(xCc,hcl.U[0].u[order:-order])

    t = 0.
    while t<Tmax:
        t = hcl.timeStepExplicit(t, Tmax)

    if plotResult:
        pl.ion()
        pl.plot(xCc,hcl.U[0].u[order:-order])
        pl.plot(xCi,V,'k:')

    return xCi, hcl.U

