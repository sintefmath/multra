import numpy as np
from copy import copy as cp

def minmod(a,b):
    return 0.5*(np.sign(a)+np.sign(b))*np.minimum(np.abs(a),np.abs(b))

def maxmod(x,y):
    return 0.5*(np.sign(x)+np.sign(y))*np.maximum(np.abs(x),np.abs(y))

def superbee(x,y):
    return maxmod(minmod(x,2.*y),minmod(2.*x,y))

def mc(x,y,z):
    dif = (((x<0.)&(y<0.)&(z<0.))!=True)&(((x>0.)&(y>0.)&(z>0.))!=True)
    x[np.nonzero(dif)]=0.0
    y[np.nonzero(dif)]=0.0
    z[np.nonzero(dif)]=0.0
    return np.sign(x)*np.minimum(np.abs(x),np.minimum(np.abs(y),np.abs(z)))

def composeLRstate(limiter, dim, order):
    if order==1:
        if dim==1:
            def lrState(u, direction, order):
                return u[0:-1], u[1:]
        else:
            def lrState(u, direction, order):
                if direction==1:
                    return u[0:-1,order:-order], u[1:,order:-order]
                else:
                    return u[order:-order,0:-1], u[order:-order,1:]
    else:
        if dim==1:
            if limiter=='minmod':
                def lrState(u, direction, order):
                    a = u[1:]-u[0:-1]
                    S = 0.5*(np.sign(a[0:-1])+np.sign(a[1:]))*np.minimum(np.abs(a[0:-1]),np.abs(a[1:])) # limiter
                    return u[1:-2] + 0.5*S[0:-1], u[2:-1] - 0.5*S[1:]
            if limiter=='superbee':
                def lrState(u, direction, order):
                    a = u[1:]-u[0:-1]
                    S = superbee(a[0:-1],a[1:]) # limiter
                    return u[1:-2] + 0.5*S[0:-1], u[2:-1] - 0.5*S[1:]
            if limiter=='mc':
                def lrState(u, direction, order):
                    a = u[1:]-u[0:-1]
                    S = mc(2.*a[0:-1],.5*(u[2:]-u[:-2]),2.*a[1:]) # limiter
                    return u[1:-2] + 0.5*S[0:-1], u[2:-1] - 0.5*S[1:]
        else:
            if limiter=='minmod':
                def lrState(u, direction, order):
                    if direction==1:
                        a = u[1:, order:-order]-u[0:-1, order:-order]
                        S = 0.5*(np.sign(a[0:-1,:])+np.sign(a[1:,:]))*np.minimum(np.abs(a[0:-1,:]),np.abs(a[1:,:])) # limiter
                        return u[1:-2, order:-order] + 0.5*S[0:-1,:], u[2:-1, order:-order] - 0.5*S[1:,:]
                    else:
                        a = u[order:-order, 1:]-u[order:-order, 0:-1]
                        S = 0.5*(np.sign(a[:,0:-1])+np.sign(a[:,1:]))*np.minimum(np.abs(a[:,0:-1]),np.abs(a[:,1:])) # limiter
                        return u[order:-order, 1:-2] + 0.5*S[:,0:-1], u[order:-order, 2:-1] - 0.5*S[:,1:]
            if limiter=='superbee':
                def lrState(u, direction, order):
                    if direction==1:
                        a = u[1:, order:-order]-u[0:-1, order:-order]
                        S = superbee(a[0:-1,:],a[1:,:]) # limiter
                        return u[1:-2, order:-order] + 0.5*S[0:-1,:], u[2:-1, order:-order] - 0.5*S[1:,:]
                    else:
                        a = u[order:-order, 1:]-u[order:-order, 0:-1]
                        S = superbee(a[:,0:-1],a[:,1:]) # limiter
                        return u[order:-order, 1:-2] + 0.5*S[:,0:-1], u[order:-order, 2:-1] - 0.5*S[:,1:]
            if limiter=='mc':
                def lrState(u, direction, order):
                    if direction==1:
                        a = u[1:, order:-order]-u[0:-1, order:-order]
                        S = mc(2.*a[0:-1,:],.5*(u[2:, order:-order]-u[:-2, order:-order]),2.*a[1:,:]) # limiter
                        return u[1:-2, order:-order] + 0.5*S[0:-1,:], u[2:-1, order:-order] - 0.5*S[1:,:]
                    else:
                        a = u[order:-order, 1:]-u[order:-order, 0:-1]
                        S = mc(2.*a[:,0:-1],.5*(u[order:-order, 2:]-u[order:-order, :-2]),2.*a[:,1:]) # limiter
                        return u[order:-order, 1:-2] + 0.5*S[:,0:-1], u[order:-order, 2:-1] - 0.5*S[:,1:]
    return lrState

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
            def boundaryCondN(U, t, dy, x):
                for i in range(len(U)):
                    U[i].u[:,-1] = U[i].u[:,-2]
        else:
            def boundaryCondN(U, t, dy, x):
                for i in range(len(U)):
                    U[i].u[:,-1] = U[i].u[:,-4]
                    U[i].u[:,-2] = U[i].u[:,-3]
    else:
        if order==1:
            def boundaryCondN(U, t, dy, x):
                uBC = bcfun(t, dy, x)
                for i in range(len(U)):
                    U[i].u[:,-1] = uBC[i]
        else:
            def boundaryCondN(U, t, dy, x):
                uBC = bcfun(t, dy, x)
                for i in range(len(U)):
                    U[i].u[:,-1] = uBC[i]
                    U[i].u[:,-2] = uBC[i]
    return boundaryCondN



class ConsQuantity:
    u = None
    u_tmp = None
    uW = None
    uS = None
    uE = None
    uN = None
    def __init__(self, nx, ny, order):
        if ny==None:
            self.u = np.zeros(nx+2*order)
        else:
            self.u = np.zeros((nx+2*order, ny+2*order))
    def savetmp(self):
        self.u_tmp = 1.*self.u


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
    boundaryCondE = None
    boundaryCondW = None
    boundaryCondN = None
    boundaryCondS = None
    boundarySource = None
    timeStepExplicit = None
    lrState = None
    fluxesSet = False
    boundaryCondSet = False

    def __init__(self, order, limiter):
        self.order = order
        self.limiter = limiter
        if order==1:
            self.timeStepExplicit = self.timeStepExplicitOrd1
        else:
            self.timeStepExplicit = self.timeStepExplicitOrd2

    def setNumericalFluxFuns(self, numFluxFunX, numFluxFunY, maxAbsEigFun):

### Checking if functions are called in the right order
        self.fluxesSet = True
###
        HyperbolicConsLaw.numFluxFunX = numFluxFunX
        HyperbolicConsLaw.numFluxFunY = numFluxFunY
        HyperbolicConsLaw.maxAbsEigFun = maxAbsEigFun

        if numFluxFunY==None:
            self.dim = 1
        else:
            self.dim = 2

    def setFluxParams(self, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def setBoundaryCond(self, boundaryCondFunE = None, boundaryCondFunW = None, boundaryCondFunN = None, boundaryCondFunS = None):

### Checking if functions are called in the right order
        assert self.fluxesSet == True, "set numerical flux functions before setting boundary conditions"
        self.boundaryCondSet = True
###

        self.lrState = composeLRstate(self.limiter, self.dim, self.order)

        self.boundaryCondE = composeBC_E(boundaryCondFunE, self.dim, self.order)
        self.boundaryCondW = composeBC_W(boundaryCondFunW, self.dim, self.order)
        if self.dim == 2:
            self.boundaryCondN = composeBC_N(boundaryCondFunN, self.order)
            self.boundaryCondS = composeBC_S(boundaryCondFunS, self.order)

    def setU(self, uinit, nx, ny, xCc, yCc):

### Checking if functions are called in the right order
        assert self.fluxesSet == True, "set numerical flux functions before setting U"
        assert self.boundaryCondSet == True, "set boundary conditions before initializing U"
###
        self.nx = nx
        self.ny = ny
        self.xCc = xCc
        self.yCc = yCc
        self.dx = xCc[1]-xCc[0]
        if self.dim==2:
            self.dy = yCc[1]-yCc[0]

        numberConservedQuantities = len(uinit)

        assert self.dim == uinit[0].ndim, "dimensions of flux functions and initial conditions are not equal"

        self.U = [ConsQuantity(self.nx, self.ny, self.order) for i in range(numberConservedQuantities)]

        for i in range(len(self.U)):
            if self.dim==1:
                self.U[i].u[self.order:-self.order] = uinit[i]
            else:
                self.U[i].u[self.order:-self.order,self.order:-self.order] = uinit[i]

    def timeStepExplicitOrd1(self, t, Tmax, CFL = 0.49):
        eig = self.maxAbsEigFun(self.U, self.dx, self.dy)
        dt = 1.*CFL/eig
        if t+dt>Tmax:
            dt=Tmax-t
        t=t+dt

        ### apply boundary conditions
        self.boundaryCondW(self.U, t, self.dx, self.yCc)
        self.boundaryCondE(self.U, t, self.dx, self.yCc)
        if self.dim==2:
            self.boundaryCondN(self.U, t, self.dy, self.xCc)
            self.boundaryCondS(self.U, t, self.dy, self.xCc)
        ### states at cell interfaces
        for i in range(len(self.U)):
            self.U[i].uW, self.U[i].uE = self.lrState(self.U[i].u, 0, self.order)
            if self.dim==2:
                self.U[i].uS, self.U[i].uN = self.lrState(self.U[i].u, 1, self.order)
        ### Fluxes across cell interfaces X-dir
        FX = self.numFluxFunX(self.U, dt, self.dx) # gives back a list
        if self.dim==1:
            for i in range(len(self.U)):
                self.U[i].u[self.order:-self.order] -= dt/self.dx*(FX[i][1:] - FX[i][0:-1])
        else:
            FY = self.numFluxFunY(self.U, dt, self.dy) # gives back a list
            for i in range(len(self.U)):
                self.U[i].u[self.order:-self.order,self.order:-self.order] -= \
                        dt/self.dx*(FX[i][:,1:] - FX[i][:,0:-1]) + \
                        dt/self.dy*(FY[i][1:,:] - FY[i][0:-1,:])

        return t

    def timeStepExplicitOrd2(self, t, Tmax, CFL = 0.49):
        eig = self.maxAbsEigFun(self.U, self.dx, self.dy)
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
                self.boundaryCondN(self.U, t, self.dy, self.xCc)
                self.boundaryCondS(self.U, t, self.dy, self.xCc)
            ### states at cell interfaces
            for i in range(len(self.U)):
                self.U[i].uW, self.U[i].uE = self.lrState(self.U[i].u, 0, self.order)
                if self.dim==2:
                    self.U[i].uS, self.U[i].uN = self.lrState(self.U[i].u, 1, self.order)
            ### Fluxes across cell interfaces X-dir
            FX = self.numFluxFunX(self.U, dt, self.dx) # gives back a list
            ### advance FV scheme
            if self.dim==1:
                for i in range(len(self.U)):
                    self.U[i].u[self.order:-self.order] -= dt/self.dx*(FX[i][1:] - FX[i][0:-1])
            else:
                FY = self.numFluxFunY(self.U, dt, self.dy) # gives back a list
                for i in range(len(self.U)):
                    self.U[i].u[self.order:-self.order,self.order:-self.order] -= \
                            dt/self.dx*(FX[i][:,1:] - FX[i][:,0:-1]) + \
                            dt/self.dy*(FY[i][1:,:] - FY[i][0:-1,:])

        for i in range(len(self.U)):
            self.U[i].u = 0.5*(self.U[i].u + self.U[i].u_tmp)

        return t

