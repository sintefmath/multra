import numpy as np
from copy import copy as cp

import types

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
                """ lr state order 1, dim 1 """
                return u[0:-1], u[1:]
        else:
            def lrState(u, direction, order):
                """ lr state order 1, dim 2 """
                if direction==1:
                    return u[0:-1,order:-order], u[1:,order:-order]
                else:
                    return u[order:-order,0:-1], u[order:-order,1:]
    else:
        if dim==1:
            if limiter=='minmod':
                def lrState(u, direction, order):
                    """ lr state order 2, limiter = minmod, dim 1 """
                    a = u[1:]-u[0:-1]
                    S = 0.5*(np.sign(a[0:-1])+np.sign(a[1:]))*np.minimum(np.abs(a[0:-1]),np.abs(a[1:])) # limiter
                    return u[1:-2] + 0.5*S[0:-1], u[2:-1] - 0.5*S[1:]
            if limiter=='superbee':
                def lrState(u, direction, order):
                    """ lr state order 2, limiter = superbee, dim 1 """
                    a = u[1:]-u[0:-1]
                    S = superbee(a[0:-1],a[1:]) # limiter
                    return u[1:-2] + 0.5*S[0:-1], u[2:-1] - 0.5*S[1:]
            if limiter=='mc':
                def lrState(u, direction, order):
                    """ lr state order 2, limiter = mc, dim 1 """
                    a = u[1:]-u[0:-1]
                    S = mc(2.*a[0:-1],.5*(u[2:]-u[:-2]),2.*a[1:]) # limiter
                    return u[1:-2] + 0.5*S[0:-1], u[2:-1] - 0.5*S[1:]
        else:
            if limiter=='minmod':
                def lrState(u, direction, order):
                    """ lr state order 2, limiter = mc, dim 2 """
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
                    """ lr state order 2, limiter = mc, dim 2 """
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
                    """ lr state order 2, limiter = mc, dim 2 """
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
    if (bcfun == None) or (bcfun == 'Neumann'):
        if dim==1:
            if order==1:
                def boundaryCondW(self, t, dx, y):
                    for i in range(self.numberConservedQuantities):
                        self.U[i].u[ 0] = self.U[i].u[ 1]
            else:
                def boundaryCondW(self, t, dx, y):
                    for i in range(self.numberConservedQuantities):
                        self.U[i].u[ 0] = self.U[i].u[ 3]
                        self.U[i].u[ 1] = self.U[i].u[ 2]
        else:
            if order==1:
                def boundaryCondW(self, t, dx, y):
                    for i in range(self.numberConservedQuantities):
                        self.U[i].u[:, 0] = self.U[i].u[:, 1]
            else:
                def boundaryCondW(self, t, dx, y):
                    for i in range(self.numberConservedQuantities):
                        self.U[i].u[:, 0] = self.U[i].u[:, 3]
                        self.U[i].u[:, 1] = self.U[i].u[:, 2]
    elif (bcfun == "Periodic") or (bcfun == "periodic"):
        if dim==1:
            if order==1:
                def boundaryCondW(self, t, dx, y):
                    for i in range(self.numberConservedQuantities):
                        self.U[i].u[ 0] = self.U[i].u[-2]
            else:
                def boundaryCondW(self, t, dx, y):
                    for i in range(self.numberConservedQuantities):
                        self.U[i].u[ 0] = self.U[i].u[-4]
                        self.U[i].u[ 1] = self.U[i].u[-3]
        else:
            if order==1:
                def boundaryCondW(self, t, dx, y):
                    for i in range(self.numberConservedQuantities):
                        self.U[i].u[:, 0] = self.U[i].u[:,-2]
            else:
                def boundaryCondW(self, t, dx, y):
                    for i in range(self.numberConservedQuantities):
                        self.U[i].u[:, 0] = self.U[i].u[:,-4]
                        self.U[i].u[:, 1] = self.U[i].u[:,-3]
    else:
        if dim==1:
            if order==1:
                def boundaryCondW(self, t, dx, y):
                    uBC = bcfun(t, dx, y)
                    for i in range(self.numberConservedQuantities):
                        self.U[i].u[ 0] = uBC[i]
            else:
                def boundaryCondW(self, t, dx, y):
                    uBC = bcfun(t, dx, y)
                    for i in range(self.numberConservedQuantities):
                        self.U[i].u[ 0] = uBC[i]
                        self.U[i].u[ 1] = uBC[i]
        else:
            if order==1:
                def boundaryCondW(self, t, dx, y):
                    uBC = bcfun(t, dx, y)
                    for i in range(self.numberConservedQuantities):
                        self.U[i].u[order:-order, 0] = uBC[i]
            else:
                def boundaryCondW(self, t, dx, y):
                    uBC = bcfun(t, dx, y)
                    for i in range(self.numberConservedQuantities):
                        self.U[i].u[order:-order, 0] = uBC[i]
                        self.U[i].u[order:-order, 1] = uBC[i]
    return boundaryCondW

def composeBC_E(bcfun, dim, order):
    if (bcfun == None) or (bcfun == 'Neumann'):
        if dim==1:
            if order==1:
                def boundaryCondE(self, t, dx, y):
                    for i in range(self.numberConservedQuantities):
                        self.U[i].u[-1] = self.U[i].u[-2]
            else:
                def boundaryCondE(self, t, dx, y):
                    for i in range(self.numberConservedQuantities):
                        self.U[i].u[-1] = self.U[i].u[-4]
                        self.U[i].u[-2] = self.U[i].u[-3]
        else:
            if order==1:
                def boundaryCondE(self, t, dx, y):
                    for i in range(self.numberConservedQuantities):
                        self.U[i].u[:, -1] = self.U[i].u[:, -2]
            else:
                def boundaryCondE(self, t, dx, y):
                    for i in range(self.numberConservedQuantities):
                        self.U[i].u[:, -1] = self.U[i].u[:, -4]
                        self.U[i].u[:, -2] = self.U[i].u[:, -3]
    elif (bcfun == "Periodic") or (bcfun == "periodic"):
        if dim==1:
            if order==1:
                def boundaryCondE(self, t, dx, y):
                    for i in range(self.numberConservedQuantities):
                        self.U[i].u[-1] = self.U[i].u[ 1]
            else:
                def boundaryCondE(self, t, dx, y):
                    for i in range(self.numberConservedQuantities):
                        self.U[i].u[-1] = self.U[i].u[ 3]
                        self.U[i].u[-2] = self.U[i].u[ 2]
        else:
            if order==1:
                def boundaryCondE(self, t, dx, y):
                    for i in range(self.numberConservedQuantities):
                        self.U[i].u[:, -1] = self.U[i].u[:,  1]
            else:
                def boundaryCondE(self, t, dx, y):
                    for i in range(self.numberConservedQuantities):
                        self.U[i].u[:, -1] = self.U[i].u[:,  3]
                        self.U[i].u[:, -2] = self.U[i].u[:,  2]
    else:
        if dim==1:
            if order==1:
                def boundaryCondE(self, t, dx, y):
                    uBC = bcfun(t, dx, y)
                    for i in range(self.numberConservedQuantities):
                        self.U[i].u[-1] = uBC[i]
            else:
                def boundaryCondE(self, t, dx, y):
                    uBC = bcfun(t, dx, y)
                    for i in range(self.numberConservedQuantities):
                        self.U[i].u[-1] = uBC[i]
                        self.U[i].u[-2] = uBC[i]
        else:
            if order==1:
                def boundaryCondE(self, t, dx, y):
                    uBC = bcfun(t, dx, y)
                    for i in range(self.numberConservedQuantities):
                        self.U[i].u[order:-order,-1] = uBC[i]
            else:
                def boundaryCondE(self, t, dx, y):
                    uBC = bcfun(t, dx, y)
                    for i in range(self.numberConservedQuantities):
                        self.U[i].u[order:-order,-1] = uBC[i]
                        self.U[i].u[order:-order,-2] = uBC[i]
    return boundaryCondE

def composeBC_S(bcfun, order):
    if (bcfun == None) or (bcfun == 'Neumann'):
        if order==1:
            def boundaryCondS(self, t, dx, y):
                for i in range(self.numberConservedQuantities):
                    self.U[i].u[0, :] = self.U[i].u[1, :]
        else:
            def boundaryCondS(self, t, dx, y):
                for i in range(self.numberConservedQuantities):
                    self.U[i].u[0, :] = self.U[i].u[3, :]
                    self.U[i].u[1, :] = self.U[i].u[2, :]
    elif (bcfun == "Periodic") or (bcfun == "periodic"):
        if order==1:
            def boundaryCondS(self, t, dx, y):
                for i in range(self.numberConservedQuantities):
                    self.U[i].u[0, :] = self.U[i].u[-2, :]
        else:
            def boundaryCondS(self, t, dx, y):
                for i in range(self.numberConservedQuantities):
                    self.U[i].u[0, :] = self.U[i].u[-4, :]
                    self.U[i].u[1, :] = self.U[i].u[-3, :]
    else:
        if order==1:
            def boundaryCondS(self, t, dx, y):
                uBC = bcfun(t, dx, y)
                for i in range(self.numberConservedQuantities):
                    self.U[i].u[0, order:-order] = uBC[i]
        else:
            def boundaryCondS(self, t, dx, y):
                uBC = bcfun(t, dx, y)
                for i in range(self.numberConservedQuantities):
                    self.U[i].u[0, order:-order] = uBC[i]
                    self.U[i].u[1, order:-order] = uBC[i]
    return boundaryCondS

def composeBC_N(bcfun, order):
    if (bcfun == None) or (bcfun == 'Neumann'):
        if order==1:
            def boundaryCondN(self, t, dy, x):
                for i in range(self.numberConservedQuantities):
                    self.U[i].u[-1, :] = self.U[i].u[-2, :]
        else:
            def boundaryCondN(self, t, dy, x):
                for i in range(self.numberConservedQuantities):
                    self.U[i].u[-1, :] = self.U[i].u[-4, :]
                    self.U[i].u[-2, :] = self.U[i].u[-3, :]
    elif (bcfun == "Periodic") or (bcfun == "periodic"):
        if order==1:
            def boundaryCondN(self, t, dy, x):
                for i in range(self.numberConservedQuantities):
                    self.U[i].u[-1, :] = self.U[i].u[ 1, :]
        else:
            def boundaryCondN(self, t, dy, x):
                for i in range(self.numberConservedQuantities):
                    self.U[i].u[-1, :] = self.U[i].u[ 3, :]
                    self.U[i].u[-2, :] = self.U[i].u[ 2, :]
    else:
        if order==1:
            def boundaryCondN(self, t, dy, x):
                uBC = bcfun(t, dy, x)
                for i in range(self.numberConservedQuantities):
                    self.U[i].u[-1, order:-order] = uBC[i]
        else:
            def boundaryCondN(self, t, dy, x):
                uBC = bcfun(t, dy, x)
                for i in range(self.numberConservedQuantities):
                    self.U[i].u[-1, order:-order] = uBC[i]
                    self.U[i].u[-2, order:-order] = uBC[i]
    return boundaryCondN



class ConsQuantity:
    u = None
    u_tmp = None
    uW = None
    uS = None
    uE = None
    uN = None
    order = None
    def __init__(self, nx, ny, order):
        self.order = order
        if ny==None:
            self.u = np.zeros(nx+2*order)
        else:
            self.u = np.zeros((ny+2*order, nx+2*order))
    def savetmp(self):
        self.u_tmp = 1.*self.u

class FluxParams:
    def setParams(self, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)


class HyperbolicConsLawNumSolver:
    U = None
    Uinit = None
    dim = None
    nx = None
    ny = None
    dx = None
    dy = None
    xCc = None
    yCc = None
    order = None
    limiter = None
    numFluxFunX = None
    numFluxFunY = None
    numSourceFun = None
    maxAbsEigFun = None
    boundaryCondE = None
    boundaryCondW = None
    boundaryCondN = None
    boundaryCondS = None

    params = None

    timeStepExplicit = None
    lrState = None
    fluxesSet = False
    boundaryCondSet = False
    initUSet = False
    numberConservedQuantities = None

    def __init__(self, dim, order, limiter, dtGiven = False, nonConsFlux = False):
        self.dim = dim
        self.order = order
        self.limiter = limiter
        self.isNonConservative = nonConsFlux
        if nonConsFlux:
            if dtGiven:
                self.timeStepExplicit = self.timeStepExplicit_Nonconservative_dtGiven
            else:
                self.timeStepExplicit = self.timeStepExplicit_Nonconservative
        else:
            if dtGiven:
                if order==1:
                    self.timeStepExplicit = self.timeStepExplicitOrd1_dtGiven
                else:
                    self.timeStepExplicit = self.timeStepExplicitOrd2_dtGiven
            else:
                if order==1:
                    self.timeStepExplicit = self.timeStepExplicitOrd1
                else:
                    self.timeStepExplicit = self.timeStepExplicitOrd2
        self.params = FluxParams()

    def setNumericalFluxFuns(self, numFluxFunX, numFluxFunY, maxAbsEigFun):

### Checking if functions are called in the right order
        self.fluxesSet = True
###
        # add methods to THIS(=self) instance of the class
        self.numFluxFunX = types.MethodType(numFluxFunX, self) 
        if numFluxFunY is not None:
            self.numFluxFunY = types.MethodType(numFluxFunY, self)
        self.maxAbsEigFun = types.MethodType(maxAbsEigFun, self)

    def setNumericalSourceFun(self, numSourceFun):
        self.numSourceFun = types.MethodType(numSourceFun, self)

    def setFluxAndSourceParams(self, **kwargs):
        self.params.setParams( **kwargs)

    def setBoundaryCond(self, boundaryCondFunE = None, boundaryCondFunW = None, boundaryCondFunN = None, boundaryCondFunS = None):

###
        self.boundaryCondSet = True
###

        self.lrState = composeLRstate(self.limiter, self.dim, self.order)

        # add methods to THIS(=self) instance of the class
        self.boundaryCondE = types.MethodType( composeBC_E(boundaryCondFunE, self.dim, self.order), self )
        self.boundaryCondW = types.MethodType( composeBC_W(boundaryCondFunW, self.dim, self.order), self )
        if self.dim == 2:
            self.boundaryCondN = types.MethodType( composeBC_N(boundaryCondFunN, self.order), self )
            self.boundaryCondS = types.MethodType( composeBC_S(boundaryCondFunS, self.order), self )

    def setUinit(self, uinit, nx, ny, xCc, yCc):

###
        self.initUSet = True
###
        self.nx = nx
        self.ny = ny
        self.xCc = xCc
        self.yCc = yCc
        self.dx = xCc[1]-xCc[0]
        if self.dim==2:
            self.dy = yCc[1]-yCc[0]

        self.numberConservedQuantities = len(uinit)

        assert self.dim == uinit[0].ndim, "Error: Dimensions of initial conditions do not match registerd dimension of problem."

        self.U = [ConsQuantity(self.nx, self.ny, self.order) for i in range(self.numberConservedQuantities)]
        self.Uinit = [ConsQuantity(self.nx, self.ny, self.order) for i in range(self.numberConservedQuantities)]

        for i in range(self.numberConservedQuantities):
            if self.dim==1:
                self.U[i].u[self.order:-self.order] = uinit[i]
                self.Uinit[i].u[self.order:-self.order] = uinit[i]
            else:
                self.U[i].u[self.order:-self.order,self.order:-self.order] = uinit[i]
                self.Uinit[i].u[self.order:-self.order,self.order:-self.order] = uinit[i]

    def selfCheck(self):
### Checking if functions are called in the right order
        assert self.boundaryCondSet == True, "Error: No boundary conditions set for U."
        assert self.initUSet == True, "Error: No initial condition set for U."
        if self.fluxesSet == False:
            print("Warning: No numerical flux functions set.")
###


    def getU(self, i):
        if self.dim==1:
            return self.U[i].u[self.order:-self.order]
        else:
            return self.U[i].u[self.order:-self.order,self.order:-self.order]

    def getUinit(self, i):
        if self.dim==1:
            return self.Uinit[i].u[self.order:-self.order]
        else:
            return self.Uinit[i].u[self.order:-self.order,self.order:-self.order]

    def resetU(self):
        if self.initUSet == False:
            print("initial condition is not set")
        else:
            for i in range(self.numberConservedQuantities):
                self.U[i].u = cp(self.Uinit[i].u)

    def timeStepExplicitOrd1(self, t, Tmax, CFL = .49):
        eig = self.maxAbsEigFun(self.U, self.dx, self.dy)
        dt = 1.*CFL/eig
        if t+dt>Tmax:
            dt=Tmax-t
        t=t+dt

        ### apply boundary conditions
        self.boundaryCondW(t-dt, self.dx, self.yCc)
        self.boundaryCondE(t-dt, self.dx, self.yCc)
        ### states at cell interfaces
        for i in range(self.numberConservedQuantities):
            self.U[i].uW, self.U[i].uE = self.lrState(self.U[i].u, 0, self.order)
        if self.dim==2:
            self.boundaryCondN(t-dt, self.dy, self.xCc)
            self.boundaryCondS(t-dt, self.dy, self.xCc)
            for i in range(self.numberConservedQuantities):
                self.U[i].uS, self.U[i].uN = self.lrState(self.U[i].u, 1, self.order)
        ### Fluxes across cell interfaces X-dir
        FX = self.numFluxFunX(self.U, dt, self.dx) # gives back a list
        if self.dim==1:
            for i in range(self.numberConservedQuantities):
                self.U[i].u[self.order:-self.order] -= dt/self.dx*(FX[i][1:] - FX[i][0:-1])
        else:
            FY = self.numFluxFunY(self.U, dt, self.dy) # gives back a list
            for i in range(self.numberConservedQuantities):
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

        for i in range(self.numberConservedQuantities):
            self.U[i].savetmp()

        for internalsteps in range(1,self.order+1):
            ### apply boundary conditions
            self.boundaryCondW(t-dt, self.dx, self.yCc)
            self.boundaryCondE(t-dt, self.dx, self.yCc)
            ### states at cell interfaces
            for i in range(self.numberConservedQuantities):
                self.U[i].uW, self.U[i].uE = self.lrState(self.U[i].u, 0, self.order)
            if self.dim==2:
                self.boundaryCondN(t-dt, self.dy, self.xCc)
                self.boundaryCondS(t-dt, self.dy, self.xCc)
                for i in range(self.numberConservedQuantities):
                    self.U[i].uS, self.U[i].uN = self.lrState(self.U[i].u, 1, self.order)
            ### Fluxes across cell interfaces X-dir
            FX = self.numFluxFunX(self.U, dt, self.dx) # gives back a list
            ### advance FV scheme
            if self.dim==1:
                for i in range(self.numberConservedQuantities):
                    self.U[i].u[self.order:-self.order] -= dt/self.dx*(FX[i][1:] - FX[i][0:-1])
            else:
                FY = self.numFluxFunY(self.U, dt, self.dy) # gives back a list
                for i in range(self.numberConservedQuantities):
                    self.U[i].u[self.order:-self.order,self.order:-self.order] -= \
                            dt/self.dx*(FX[i][:,1:] - FX[i][:,0:-1]) + \
                            dt/self.dy*(FY[i][1:,:] - FY[i][0:-1,:])

        for i in range(self.numberConservedQuantities):
            self.U[i].u = 0.5*(self.U[i].u + self.U[i].u_tmp)

        return t

    def timeStepExplicitOrd1_dtGiven(self, t, dt):
        t=t+dt

        ### apply boundary conditions
        self.boundaryCondW(t-dt, self.dx, self.yCc)
        self.boundaryCondE(t-dt, self.dx, self.yCc)
        ### states at cell interfaces
        for i in range(self.numberConservedQuantities):
            self.U[i].uW, self.U[i].uE = self.lrState(self.U[i].u, 0, self.order)
        if self.dim==2:
            self.boundaryCondN(t-dt, self.dy, self.xCc)
            self.boundaryCondS(t-dt, self.dy, self.xCc)
            for i in range(self.numberConservedQuantities):
                self.U[i].uS, self.U[i].uN = self.lrState(self.U[i].u, 1, self.order)
        ### Fluxes across cell interfaces X-dir
        FX = self.numFluxFunX(self.U, dt, self.dx) # gives back a list
        if self.dim==1:
            for i in range(self.numberConservedQuantities):
                self.U[i].u[self.order:-self.order] -= dt/self.dx*(FX[i][1:] - FX[i][0:-1])
        else:
            FY = self.numFluxFunY(self.U, dt, self.dy) # gives back a list
            for i in range(self.numberConservedQuantities):
                self.U[i].u[self.order:-self.order,self.order:-self.order] -= \
                        dt/self.dx*(FX[i][:,1:] - FX[i][:,0:-1]) + \
                        dt/self.dy*(FY[i][1:,:] - FY[i][0:-1,:])

        return t

    def timeStepExplicitOrd2_dtGiven(self, t, dt):
        t=t+dt

        for i in range(self.numberConservedQuantities):
            self.U[i].savetmp()

        for internalsteps in range(1,self.order+1):
            ### apply boundary conditions
            self.boundaryCondW(t-dt, self.dx, self.yCc)
            self.boundaryCondE(t-dt, self.dx, self.yCc)
            ### states at cell interfaces
            for i in range(self.numberConservedQuantities):
                self.U[i].uW, self.U[i].uE = self.lrState(self.U[i].u, 0, self.order)
            if self.dim==2:
                self.boundaryCondN(t-dt, self.dy, self.xCc)
                self.boundaryCondS(t-dt, self.dy, self.xCc)
                for i in range(self.numberConservedQuantities):
                    self.U[i].uS, self.U[i].uN = self.lrState(self.U[i].u, 1, self.order)
            ### Fluxes across cell interfaces X-dir
            FX = self.numFluxFunX(self.U, dt, self.dx) # gives back a list
            ### advance FV scheme
            if self.dim==1:
                for i in range(self.numberConservedQuantities):
                    self.U[i].u[self.order:-self.order] -= dt/self.dx*(FX[i][1:] - FX[i][0:-1])
            else:
                FY = self.numFluxFunY(self.U, dt, self.dy) # gives back a list
                for i in range(self.numberConservedQuantities):
                    self.U[i].u[self.order:-self.order,self.order:-self.order] -= \
                            dt/self.dx*(FX[i][:,1:] - FX[i][:,0:-1]) + \
                            dt/self.dy*(FY[i][1:,:] - FY[i][0:-1,:])

        for i in range(self.numberConservedQuantities):
            self.U[i].u = 0.5*(self.U[i].u + self.U[i].u_tmp)

        return t


    def timeStepExplicit_Nonconservative(self, t, Tmax, CFL = .49):
        eig = self.maxAbsEigFun(self.U, self.dx, self.dy)
        dt = 1.*CFL/eig
        if t+dt>Tmax:
            dt=Tmax-t
        t=t+dt

        ### apply boundary conditions
        self.boundaryCondW(t-dt, self.dx, self.yCc)
        self.boundaryCondE(t-dt, self.dx, self.yCc)
        ### states at cell interfaces
        for i in range(self.numberConservedQuantities):
            self.U[i].uW, self.U[i].uE = self.lrState(self.U[i].u, 0, self.order)
        if self.dim==2:
            self.boundaryCondN(t-dt, self.dy, self.xCc)
            self.boundaryCondS(t-dt, self.dy, self.xCc)
            for i in range(self.numberConservedQuantities):
                self.U[i].uS, self.U[i].uN = self.lrState(self.U[i].u, 1, self.order)
        ### Fluxes
        FX = self.numFluxFunX(self.U, dt, self.dx) # gives back a list
        if self.numSourceFun == None:
            if self.dim==1:
                for i in range(self.numberConservedQuantities):
                    self.U[i].u[self.order:-self.order] -= dt*FX[i]/self.dx
            else:
        ### Fluxes
                FY = self.numFluxFunY(self.U, dt, self.dy) # gives back a list
                for i in range(self.numberConservedQuantities):
                    self.U[i].u[self.order:-self.order,self.order:-self.order] -= dt*(FX[i]/self.dx + FY[i]/self.dy)
        else:
        ### Sources
            S = self.numSourceFun(self.U, dt, self.dx, self.dy)
            if self.dim==1:
                for i in range(self.numberConservedQuantities):
                    self.U[i].u[self.order:-self.order] -= dt*(FX[i]/self.dx - S[i])
            else:
        ### Fluxes
                FY = self.numFluxFunY(self.U, dt, self.dy) # gives back a list
                for i in range(self.numberConservedQuantities):
                    self.U[i].u[self.order:-self.order,self.order:-self.order] -= dt*(FX[i]/self.dx + FY[i]/self.dy - S[i])

        return t

    def timeStepExplicit_Nonconservative_dtGiven(self, t, dt):
        t=t+dt

        ### apply boundary conditions
        self.boundaryCondW(t-dt, self.dx, self.yCc)
        self.boundaryCondE(t-dt, self.dx, self.yCc)
        ### states at cell interfaces
        for i in range(self.numberConservedQuantities):
            self.U[i].uW, self.U[i].uE = self.lrState(self.U[i].u, 0, self.order)
        if self.dim==2:
            self.boundaryCondN(t-dt, self.dy, self.xCc)
            self.boundaryCondS(t-dt, self.dy, self.xCc)
            for i in range(self.numberConservedQuantities):
                self.U[i].uS, self.U[i].uN = self.lrState(self.U[i].u, 1, self.order)
        ### Fluxes
        FX = self.numFluxFunX(self.U, dt, self.dx) # gives back a list
        if self.numSourceFun == None:
            if self.dim==1:
                for i in range(self.numberConservedQuantities):
                    self.U[i].u[self.order:-self.order] -= dt*FX[i]/self.dx
            else:
        ### Fluxes
                FY = self.numFluxFunY(self.U, dt, self.dy) # gives back a list
                for i in range(self.numberConservedQuantities):
                    self.U[i].u[self.order:-self.order,self.order:-self.order] -= dt*(FX[i]/self.dx + FY[i]/self.dy)
        else:
        ### Sources
            S = self.numSourceFun(self.U, dt, self.dx, self.dy)
            if self.dim==1:
                for i in range(self.numberConservedQuantities):
                    self.U[i].u[self.order:-self.order] -= dt*(FX[i]/self.dx - S[i])
            else:
        ### Fluxes
                FY = self.numFluxFunY(self.U, dt, self.dy) # gives back a list
                for i in range(self.numberConservedQuantities):
                    self.U[i].u[self.order:-self.order,self.order:-self.order] -= dt*(FX[i]/self.dx + FY[i]/self.dy - S[i])

        return t

def apply_BC_W(u, bcfun, dim, order):
    if (bcfun == None) or (bcfun == 'Neumann'):
        if dim==1:
            if order==1:
                u[ 0] = u[ 1]
            else:
                u[ 0] = u[ 3]
                u[ 1] = u[ 2]
        else:
            if order==1:
                u[:, 0] = u[:, 1]
            else:
                u[:, 0] = u[:, 3]
                u[:, 1] = u[:, 2]
    elif (bcfun == "Periodic") or (bcfun == "periodic"):
        if dim==1:
            if order==1:
                u[ 0] = u[-2]
            else:
                u[ 0] = u[-4]
                u[ 1] = u[-3]
        else:
            if order==1:
                u[:, 0] = u[:,-2]
            else:
                u[:, 0] = u[:,-4]
                u[:, 1] = u[:,-3]
    return u

def apply_BC_E(u, bcfun, dim, order):
    if (bcfun == None) or (bcfun == 'Neumann'):
        if dim==1:
            if order==1:
                u[-1] = u[-2]
            else:
                u[-1] = u[-4]
                u[-2] = u[-3]
        else:
            if order==1:
                u[:, -1] = u[:, -2]
            else:
                u[:, -1] = u[:, -4]
                u[:, -2] = u[:, -3]
    elif (bcfun == "Periodic") or (bcfun == "periodic"):
        if dim==1:
            if order==1:
                u[-1] = u[ 1]
            else:
                u[-1] = u[ 3]
                u[-2] = u[ 2]
        else:
            if order==1:
                u[:, -1] = u[:,  1]
            else:
                u[:, -1] = u[:,  3]
                u[:, -2] = u[:,  2]
    return u

def apply_BC_S(u, bcfun, order):
    if (bcfun == None) or (bcfun == 'Neumann'):
        if order==1:
            u[0, :] = u[1, :]
        else:
            u[0, :] = u[3, :]
            u[1, :] = u[2, :]
    elif (bcfun == "Periodic") or (bcfun == "periodic"):
        if order==1:
            u[0, :] = u[-2, :]
        else:
            u[0, :] = u[-4, :]
            u[1, :] = u[-3, :]
    return u

def apply_BC_N(u, bcfun, order):
    if (bcfun == None) or (bcfun == 'Neumann'):
        if order==1:
            u[-1, :] = u[-2, :]
        else:
            u[-1, :] = u[-4, :]
            u[-2, :] = u[-3, :]
    elif (bcfun == "Periodic") or (bcfun == "periodic"):
        if order==1:
            u[-1, :] = u[ 1, :]
        else:
            u[-1, :] = u[ 3, :]
            u[-2, :] = u[ 2, :]
    return u
