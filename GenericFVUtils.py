import numpy as np
from copy import copy as cp

def minmod(a,b):
    return 0.5*(np.sign(a)+np.sign(b))*np.minimum(np.abs(a),np.abs(b))

def maxmod(x,y):
    return 0.5*(np.sign(x)+np.sign(y))*np.maximum(np.abs(x),np.abs(y))

def superbee(x,y):
    return maxmod(minmod(x,2*y),minmod(2*x,y))

def mc(x,y,z):
    dif = (((x<0.)&(y<0.)&(z<0.))!=True)&(((x>0.)&(y>0.)&(z>0.))!=True)
    x[np.nonzero(dif)]=0.0
    y[np.nonzero(dif)]=0.0
    z[np.nonzero(dif)]=0.0
    return np.sign(x)*np.minimum(np.abs(x),np.minimum(np.abs(y),np.abs(z)))

def getSlopeMMDim1(u, direction):
    a = u[1:]-u[0:-1]
    return minmod(a[0:-1],a[1:])
def getSlopeMMDim2(u, direction):
    if direction==1:
        a = u[1:,:]-u[0:-1,:]
        S = minmod(a[0:-1,:],a[1:,:])
    else:
        a = u[:,1:]-u[:,0:-1]
        S = minmod(a[:,0:-1],a[:,1:])
    return S
def getSlopeSPDim1(u, direction):
    a = u[1:]-u[0:-1]
    return superbee(a[0:-1],a[1:])
def getSlopeSPDim2(u, direction):
    if direction==1:
        a = u[1:,:]-u[0:-1,:]
        S = superbee(a[0:-1,:],a[1:,:])
    else:
        a = u[:,1:]-u[:,0:-1]
        S = superbee(a[:,0:-1],a[:,1:])
    return S
def getSlopeMCDim1(u, direction):
    a = u[1:]-u[0:-1]
    return mc(2.*a[0:-1],.5*(u[2:]-u[:-2]),2.*a[1:])
def getSlopeMCDim2(u, direction):
    if direction==1:
        a = u[1:,:]-u[0:-1,:]
        S = mc(2.*a[0:-1,:],.5*(u[2:,:]-u[:-2,:]),2.*a[1:,:])
    else:
        a = u[:,1:]-u[:,0:-1]
        S = mc(2.*a[:,0:-1],.5*(u[:,2:]-u[:,:-2]),2.*a[:,1:])
    return S

def composeBC_W(bcfun, dim, order):
    if bcfun == None:
        if dim==1:
            if order==1:
                def boundaryCondW(U, t, dx, y):
                    for i in range(len(U)):
                        U[i].u[ 0] = U[i].u[ 1]
            else:
                def boundaryCondW(U, t, dx, y):
                    for i in range(len(U)):
                        U[i].u[ 0] = U[i].u[ 3]
                        U[i].u[ 1] = U[i].u[ 2]
        else:
            if order==1:
                def boundaryCondW(U, t, dx, y):
                    for i in range(len(U)):
                        U[i].u[ 0,:] = U[i].u[ 1,:]
            else:
                def boundaryCondW(U, t, dx, y):
                    for i in range(len(U)):
                        U[i].u[ 0,:] = U[i].u[ 3,:]
                        U[i].u[ 1,:] = U[i].u[ 2,:]
    else:
        if dim==1:
            if order==1:
                def boundaryCondW(U, t, dx, y):
                    uBC = bcfun(t, dx, y)
                    for i in range(len(U)):
                        U[i].u[ 0] = uBC[i]
            else:
                def boundaryCondW(U, t, dx, y):
                    uBC = bcfun(t, dx, y)
                    for i in range(len(U)):
                        U[i].u[ 0] = uBC[i]
                        U[i].u[ 1] = uBC[i]
        else:
            if order==1:
                def boundaryCondW(U, t, dx, y):
                    uBC = bcfun(t, dx, y)
                    for i in range(len(U)):
                        U[i].u[ 0,:] = uBC[i]
            else:
                def boundaryCondW(U, t, dx, y):
                    uBC = bcfun(t, dx, y)
                    for i in range(len(U)):
                        U[i].u[ 0,:] = uBC[i]
                        U[i].u[ 1,:] = uBC[i]
    return boundaryCondW

def composeBC_E(bcfun, dim, order):
    if bcfun == None:
        if dim==1:
            if order==1:
                def boundaryCondE(U, t, dx, y):
                    for i in range(len(U)):
                        U[i].u[-1] = U[i].u[-2]
            else:
                def boundaryCondE(U, t, dx, y):
                    for i in range(len(U)):
                        U[i].u[-1] = U[i].u[-4]
                        U[i].u[-2] = U[i].u[-3]
        else:
            if order==1:
                def boundaryCondE(U, t, dx, y):
                    for i in range(len(U)):
                        U[i].u[-1,:] = U[i].u[-2,:]
            else:
                def boundaryCondE(U, t, dx, y):
                    for i in range(len(U)):
                        U[i].u[-1,:] = U[i].u[-4,:]
                        U[i].u[-2,:] = U[i].u[-3,:]
    else:
        if dim==1:
            if order==1:
                def boundaryCondE(U, t, dx, y):
                    uBC = bcfun(t, dx, y)
                    for i in range(len(U)):
                        U[i].u[-1] = uBC[i]
            else:
                def boundaryCondE(U, t, dx, y):
                    uBC = bcfun(t, dx, y)
                    for i in range(len(U)):
                        U[i].u[-1] = uBC[i]
                        U[i].u[-2] = uBC[i]
        else:
            if order==1:
                def boundaryCondE(U, t, dx, y):
                    uBC = bcfun(t, dx, y)
                    for i in range(len(U)):
                        U[i].u[-1,:] = uBC[i]
            else:
                def boundaryCondE(U, t, dx, y):
                    uBC = bcfun(t, dx, y)
                    for i in range(len(U)):
                        U[i].u[-1,:] = uBC[i]
                        U[i].u[-2,:] = uBC[i]
    return boundaryCondE

def composeBC_S(bcfun, order):
    if bcfun == None:
        if order==1:
            def boundaryCondS(U, t, dx, y):
                for i in range(len(U)):
                    U[i].u[:, 0] = U[i].u[:, 1]
        else:
            def boundaryCondS(U, t, dx, y):
                for i in range(len(U)):
                    U[i].u[:, 0] = U[i].u[:, 3]
                    U[i].u[:, 1] = U[i].u[:, 2]
    else:
        if order==1:
            def boundaryCondS(U, t, dx, y):
                uBC = bcfun(t, dx, y)
                for i in range(len(U)):
                    U[i].u[:, 0] = uBC[i]
        else:
            def boundaryCondS(U, t, dx, y):
                uBC = bcfun(t, dx, y)
                for i in range(len(U)):
                    U[i].u[:, 0] = uBC[i]
                    U[i].u[:, 1] = uBC[i]
    return boundaryCondS

def composeBC_N(bcfun, order):
    if bcfun == None:
        if order==1:
            def boundaryCondN(U, t, dx, y):
                for i in range(len(U)):
                    U[i].u[:,-1] = U[i].u[:,-2]
        else:
            def boundaryCondN(U, t, dx, y):
                for i in range(len(U)):
                    U[i].u[:,-1] = U[i].u[:,-4]
                    U[i].u[:,-2] = U[i].u[:,-3]
    else:
        if order==1:
            def boundaryCondN(U, t, dx, y):
                uBC = bcfun(t, dx, y)
                for i in range(len(U)):
                    U[i].u[:,-1] = uBC[i]
        else:
            def boundaryCondN(U, t, dx, y):
                uBC = bcfun(t, dx, y)
                for i in range(len(U)):
                    U[i].u[:,-1] = uBC[i]
                    U[i].u[:,-2] = uBC[i]
    return boundaryCondN



class ConsQuantity:
    u = None
    u_tmp = None
    numFx = None
    numFy = None
    uW = None
    uS = None
    uE = None
    uN = None
    getSlope = None
    def initialize(self, uinit):
        self.u = uinit
    def savetmp(self):
        self.u_tmp = 1.*self.u

class ConsQuantityOrd1Dim1(ConsQuantity):
    def initialize(self, limiter, uinit):
        super().initialize(uinit)
    def LRstate(self, direction):
        uL = self.u[0:-1]
        uR = self.u[1:]
        return uL, uR

class ConsQuantityOrd1Dim2(ConsQuantity):
    def initialize(self, limiter, uinit):
        super().initialize(uinit)
    def LRstate(self, direction):
        if direction==1:
            uL = self.u[0:-1,:]
            uR = self.u[1:,:]
        else:
            uL = self.u[:,0:-1]
            uR = self.u[:,1:]
        return uL, uR

class ConsQuantityOrd2Dim1(ConsQuantity):
    def initialize(self, limiter, uinit):
        super().initialize(uinit)
        if limiter=='minmod':
            self.getSlope = getSlopeMMDim1
        if limiter=='superbee':
            self.getSlope = getSlopeSPDim1
        if limiter=='mc':
            self.getSlope = getSlopeMCDim1
    def LRstate(self, direction):
        S = self.getSlope(self.u,direction)
        uL = self.u[1:-2] + 0.5*S[0:-1]
        uR = self.u[2:-1] - 0.5*S[1:]
        return uL, uR

class ConsQuantityOrd2Dim2(ConsQuantity):
    def initialize(self, limiter, uinit):
        super().initialize(uinit)
        if limiter=='minmod':
            self.getSlope = getSlopeMMDim2
        if limiter=='superbee':
            self.getSlope = getSlopeSPDim2
        if limiter=='mc':
            self.getSlope = getSlopeMCDim2
    def LRstate(self, direction):
        if direction==1:
            S = self.getSlope(self.u,direction)
            uL = self.u[1:-2,:] + 0.5*S[0:-1,:]
            uR = self.u[2:-1,:] - 0.5*S[1:,:]
        else:
            S = self.getSlope(self.u,direction)
            uL = self.u[:,1:-2] + 0.5*S[:,0:-1]
            uR = self.u[:,2:-1] - 0.5*S[:,1:]
        return uL, uR

class HyperbolicConsLaw:
    U = None
    dim = None
    nx = None
    ny = None
    dx = None
    dy = None
    xCc = None
    yCc = None
    order = None
    limiter = None
    maxEigFun = None
    numFluxFunX = None
    numFluxFunY = None
    boundaryCondE = None
    boundaryCondW = None
    boundaryCondN = None
    boundaryCondS = None
    boundaryCondFunE = None
    boundaryCondFunW = None
    boundaryCondFunN = None
    boundaryCondFunS = None
    boundarySource = None

    def setFuns(self, maxEigFun, numFluxFunX, numFluxFunY, boundaryCondFunE, boundaryCondFunW, boundaryCondFunN, boundaryCondFunS, order, limiter):
        self.maxEigFun = maxEigFun
        self.numFluxFunX = numFluxFunX
        self.numFluxFunY = numFluxFunY

        self.boundaryCondFunE = boundaryCondFunE
        self.boundaryCondFunW = boundaryCondFunW
        self.boundaryCondFunN = boundaryCondFunN
        self.boundaryCondFunS = boundaryCondFunS

        self.order = order
        self.limiter = limiter

    def setU(self, uinit, nx, ny, xCc, yCc):
        self.nx = nx
        self.ny = ny
        self.xCc = xCc
        self.yCc = yCc
        self.dx = xCc[1]-xCc[0]
        self.dy = yCc[1]-yCc[0]

        numberConservedQuantities = len(uinit)

        self.dim = uinit[0].ndim
        
        self.boundaryCondE = composeBC_E(self.boundaryCondFunE, self.dim, self.order)
        self.boundaryCondW = composeBC_W(self.boundaryCondFunW, self.dim, self.order)
        if self.dim == 2:
            self.boundaryCondN = composeBC_N(self.boundaryCondFunN, self.dim, self.order)
            self.boundaryCondS = composeBC_S(self.boundaryCondFunS, self.dim, self.order)

        if self.dim == 1:
            if self.order==1:
                self.U = [ConsQuantityOrd1Dim1() for i in range(numberConservedQuantities)]
            else:
                self.U = [ConsQuantityOrd2Dim1() for i in range(numberConservedQuantities)]
        if self.dim == 2:
            if self.order==1:
                self.U = [ConsQuantityOrd1Dim2() for i in range(numberConservedQuantities)]
            else:
                self.U = [ConsQuantityOrd2Dim2() for i in range(numberConservedQuantities)]

        for i in range(len(self.U)):
            self.U[i].initialize(self.limiter, uinit[i])

class HyperbolicConsLawOrd1(HyperbolicConsLaw):
    def timeStepExplicit(self, t, Tmax, CFL = 0.49):
        eig = self.maxEigFun(self.U, self.dx, self.dy)
        dt = 1.*CFL/eig
        if t+dt>Tmax:
            dt=Tmax-t
        t=t+dt

        ### apply boundary conditions
        self.boundaryCondW(self.U, t, self.dx, self.yCc)
        self.boundaryCondE(self.U, t, self.dx, self.yCc)
        if self.dim==2:
            self.boundaryCondN(self.U, self.dy, self.xCc)
            self.boundaryCondS(self.U, self.dy, self.xCc)
        ### states at cell interfaces
        for i in range(len(self.U)):
            self.U[i].uW, self.U[i].uE = self.U[i].LRstate(1)
            if self.dim==2:
                self.U[i].uS, self.U[i].uN = self.U[i].LRstate(2)
        ### Fluxes across cell interfaces X-dir
        FX = self.numFluxFunX(self.U, dt, self.dx) # gives back a list
        if self.dim==2:
            FY = self.numFluxFunY(self.U, dt, self.dy) # gives back a list
        ### advance FV scheme
        for i in range(len(self.U)):
            if self.dim==1:
                self.U[i].u[self.order:-self.order] -= dt/self.dx*(FX[i][1:] - FX[i][0:-1])
            else:
                self.U[i].u[self.order:-self.order,:] -= dt/self.dx*(FX[i][1:,:] - FX[i][0:-1,:])
                self.U[i].u[:,self.order:-self.order] -= dt/self.dy*(FY[i][:,1:] - FY[i][:,0:-1])

        return t

class HyperbolicConsLawOrd2(HyperbolicConsLaw):
    def timeStepExplicit(self, t, Tmax, CFL = 0.49):
        #while t<Tinterval[1]:
        eig = self.maxEigFun(self.U, self.dx, self.dy)
        dt = 1.*CFL/eig
        if t+dt>Tmax:
            dt=Tmax-t
        t=t+dt

        for i in range(len(self.U)):
            self.U[i].savetmp()

        for internalsteps in range(1,self.order+1):
            ### apply boundary conditions
            self.boundaryCondW(self.U, t, self.dx, self.yCc)
            self.boundaryCondE(self.U, t, self.dx, self.yCc)
            if self.dim==2:
                self.boundaryCondN(self.U, self.dy, self.xCc)
                self.boundaryCondS(self.U, self.dy, self.xCc)
            ### states at cell interfaces
            for i in range(len(self.U)):
                self.U[i].uW, self.U[i].uE = self.U[i].LRstate(1)
                if self.dim==2:
                    self.U[i].uS, self.U[i].uN = self.U[i].LRstate(2)
            ### Fluxes across cell interfaces X-dir
            FX = self.numFluxFunX(self.U, dt, self.dx) # gives back a list
            if self.dim==2:
                FY = self.numFluxFunY(self.U, dt, self.dy) # gives back a list
            ### advance FV scheme
            for i in range(len(self.U)):
                if self.dim==1:
                    self.U[i].u[self.order:-self.order] -= dt/self.dx*(FX[i][1:] - FX[i][0:-1])
                else:
                    self.U[i].u[self.order:-self.order,:] -= dt/self.dx*(FX[i][1:] - FX[i][0:-1])
                    self.U[i].u[:,self.order:-self.order] -= dt/self.dy*(FY[i][1:] - FY[i][0:-1])

        for i in range(len(self.U)):
            self.U[i].u = 0.5*(self.U[i].u + self.U[i].u_tmp)

        return t

