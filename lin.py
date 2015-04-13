import pylab as pl
from GenericFVUtils import *

a = None

def maxEig(U, dx, dy):
    global a
    return np.max(np.abs(a))/dx

def FluxX(u, m):
    global a
    return a[m]*u[m]

def numFluxX_LxF(U, dt, dx):
    return [0.5*(FluxX(U[0].uW) + FluxX(U[0].uE)) - 0.5*dx/dt*(U[0].uE - U[0].uW)]

def numFluxX_upwind(U, dt, dx):
    global a
#
#    ap = ((a>=0.)==True)
#    am = ((a<0.)==True)
#    Fu = np.zeros_like(U[0].uW)
#    Fu[np.nonzero(ap)] = FluxX(U[0].uW)[np.nonzero(ap)]
#    Fu[np.nonzero(am)] = FluxX(U[0].uE)[np.nonzero(am)]
#    return [Fu]
#
    mask = a < 0.
    Fu = np.empty_like(U[0].uW)
    Fu[-mask] = FluxX(U[0].uW,-mask)
    Fu[mask] = FluxX(U[0].uE,mask)
    return [Fu]

def boundaryCondFunE(t, dx, y):
    return [0.]
def boundaryCondFunW(t, dx, y):
    u = 0.
    x0 = 0.4
    if (t<x0):
        u = 0.25
    return [u]

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

    uinit = np.zeros((nx+2*order))
    #uinit[order:-order] = np.sin(2*np.pi*xCc)

    hcl.setU([uinit], nx, ny, xCc, yCc)

    global a
    a = -xCi + .5

    #pl.plot(xCc,hcl.U[0].u[order:-order])

    t = 0.
    while t<Tmax:
        t = hcl.timeStepExplicit(t, Tmax, dx)

    if plotResult:
        pl.ion()
        pl.plot(xCc,hcl.U[0].u[order:-order])
        pl.plot(xCi,a,'k:')

    return xCi, hcl.U

