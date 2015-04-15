import pylab as pl
from GenericFVUtils import *

a = None
b = None

def maxEig(U, dx, dy):
    global a
    global b
    return max(np.max(np.abs(a))/dx,np.max(np.abs(b))/dy)

def FluxX(u, m):
    global a
    return a[m]*u[m]

def FluxY(u, m):
    global b
    return b[m]*u[m]

def numFluxX_upwind(U, dt, dx):
    global a
    mask = a < 0.
    Fu = np.empty_like(U[0].uW)
    Fu[-mask] = FluxX(U[0].uW,-mask)
    Fu[mask] = FluxX(U[0].uE,mask)
    return [Fu]

def numFluxY_upwind(U, dt, dx):
    global b
    mask = b < 0.
    Fu = np.empty_like(U[0].uS)
    Fu[-mask] = FluxY(U[0].uS,-mask)
    Fu[mask] = FluxY(U[0].uN,mask)
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


def linear(nx=100, ny=100 ,Tmax=1., order=1, limiter='minmod', plotResult = True):

    hcl = HyperbolicConsLaw()

    xCi = np.linspace(0,1,nx+1) # cell interfaces
    dx = xCi[1]-xCi[0]

    yCi = np.linspace(0,1,ny+1) # cell interfaces
    dy = yCi[1]-yCi[0]

    boundaryCondFunN = None
    boundaryCondFunS = None
    boundaryCondFunW = None
    boundaryCondFunE = None

    hcl.setFuns(maxEig, numFluxX_upwind, numFluxY_upwind, boundaryCondFunE, boundaryCondFunW, boundaryCondFunN, boundaryCondFunS, order, limiter)

    xCc = np.linspace(0.+dx/2.,1.-dx/2.,nx) # cell centers
    yCc = np.linspace(0.+dy/2.,1.-dy/2.,ny) # cell centers

    xv, yv = np.meshgrid(xCc, yCc)

    uinit = np.zeros((ny, nx))
    #uinit[np.nonzero((xv<.75) & (xv>.25))] = 1.
    #uinit[np.nonzero((yv<.75) & (yv>.25))] = 1.
    uinit[np.sqrt((xv-.5)**2 + (yv-.5)**2)<.25] = 1.

    hcl.setU([uinit], ny, nx, xCc, yCc)

    #pl.ion()
    #pl.imshow(hcl.U[0].u[order:-order,order:-order])
    #return

    global a
    a = .5*np.ones((ny,nx+1))
    global b
    b = .5*np.ones((ny+1,nx))

    t = 0.
    while t<Tmax:
        t = hcl.timeStepExplicit(t, Tmax)

    if plotResult:
        pl.ion()
        #pl.imshow(xCi, yCi, hcl.U[0].u[order:-order,order:-order], interpolation='nearest', cmap=pl.gray(), origin='lower')
        pl.pcolor(xv, yv, hcl.U[0].u[order:-order,order:-order], cmap=pl.gray())

    return xCi, yCi, hcl.U

