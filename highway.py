import numpy as np
from copy import copy as cp

import matplotlib.pyplot as pl

import time

count=0

def draw(roads, t, o, save):
    pl.figure(1)
    pl.clf()
    for i in range(len(roads)):
        x=np.linspace(roads[i].geo_IN[0], roads[i].geo_OUT[0], roads[i].n)
        pl.plot(x, roads[i].rho[o:-o]);
    pl.ylim([0,1.1])
    pl.title("t = "+str(t))
    if save:
        pl.savefig("sim_"+str(int(t)).zfill(4))
    else:
        pl.show()



def slope(u,limiter):
    a = u[1:]-u[0:-1]
    if limiter=='minmod':
        S = minmod(a[0:-1],a[1:])
    if limiter=='superbee':
        S = superbee(a[0:-1],a[1:])
    if limiter=='mc':
        S = mc(2.*a[0:-1],.5*(u[2:]-u[:-2]),2.*a[1:])
    return S


class Road:

    x_IN = None
    x_OUT = None
    n = None
    dx = None

    rho = None
    q = None

    rho_tmp = None
    q_tmp = None

    Umax = None
    roadIn = None
    roadOut = None

    def rhoLR(self, o, limiter):
        if o==1:
            rhoL = self.rho[0:-1]
            rhoR = self.rho[1:]
        if o==2:
            S = slope(self.rho,limiter)
            rhoL = self.rho[1:-2] + 0.5*S[0:-1]
            rhoR = self.rho[2:-1] - 0.5*S[1:]
        return rhoL, rhoR

    def maxEig(self):
        DF = Flux_drho(self.rho, self.Umax, self.rhomax)
        return max(np.abs(DF))

    def maketmp(self):
        self.rho_tmp = 1.*self.rho
        self.q_tmp = 1.*self.q

    def initialize(self, n, geo_IN, geo_OUT, o, Umax=1., rhomax=1., rhoinit=0.):
        self.rho = rhoinit + np.zeros((n+2*o,1))
        self.q = np.zeros((n+2*o,1))
        self.Umax = Umax
        self.rhomax = rhomax
        self.geo_IN = geo_IN
        self.geo_OUT = geo_OUT
        self.dx = np.linalg.norm(geo_IN - geo_OUT)/(1.*n)
        self.n = n

class Intersections:

    roads_IN = None
    roads_OUT = None

    distribution = None

    fixedRedTimes = None

    def initialize(self, roads_IN, roads_OUT, distribution=None, fixedRedTimes=None):

        self.roads_IN = roads_IN
        self.roads_OUT = roads_OUT
        self.distribution = distribution
        self.fixedRedTimes = fixedRedTimes

    def applyBC(self, roads, o, t):
        if self.roads_IN<0:
            for i in self.roads_OUT:
                roads[i].rho = BC_inflow(roads[i].rho,o,t)
        elif self.roads_OUT<0:
            for i in self.roads_IN:
                roads[i].rho = BC_outflow(roads[i].rho,o,t)
        else:
            redLight = False
            if self.fixedRedTimes:
                for i in range(int(self.fixedRedTimes.size/2)):
                    if (self.fixedRedTimes[2*i]<=t and t<self.fixedRedTimes[2*i+1]):
                        redLight = True
                        break
            if redLight:
                for r in self.roads_OUT:
                    for i in range(1,o+1):
                        roads[r].rho[o-i] = 0.
                for r in self.roads_IN:
                    for i in range(1,o+1):
                        roads[r].rho[-o+i-1] = 1.
            else:
                #roads[self.roads_OUT].rho[o-1]   = self.distribution*roads[self.roads_IN].rho[-o-i]
                #roads[self.roads_IN].rho[-o+i-1] = self.distribution*roads[self.roads_OUT].rho[o+i-1]
                for rOut in self.roads_OUT:
                    for rIn in self.roads_IN:
                        for i in range(1,o+1):
                            roads[rOut].rho[o-i] = roads[rIn].rho[-o-i]
                            roads[rIn].rho[-o+i-1] = roads[rOut].rho[o+i-1]

def Flux(rho, Umax, rhomax):
    return rho*Umax*(1.-rho/rhomax)

def Flux_drho(rho, Umax, rhomax):
    return Umax*(1.-2.*rho/rhomax)

def demand(rho, Umax, rhomax):
    if rho <= 0.5*rhomax:
        return Flux(rho, Umax, rhomax)
    else:
        return Flux(0.5*rhomax, Umax, rhomax)

def supply(rho, Umax, rhomax):
    if rho <= 0.5*rhomax:
        return Flux(0.5*rhomax, Umax, rhomax)
    else:
        return Flux(rho, Umax, rhomax)


def ConsLaw(roads, intersections, Tinterval, limiter, solver, o):

    t = cp(Tinterval[0])

    while t<Tinterval[1]:

        eig = roads[0].maxEig()/roads[0].dx
        for i in range(1,len(roads)):
            eig = max(eig,roads[i].maxEig()/roads[i].dx)

        if (o==1):
            CFL = 0.99
        if (o==2):
            CFL = 0.49

        dt = 1.*CFL/eig

        if t+dt>Tinterval[1]:
            dt=Tinterval[1]-t
        t=t+dt

        if o==2:
            for i in range(len(roads)):
                roads[i].maketmp()

        for internalsteps in range(1,o+1):
            ### apply boundary conditions for all roads
            for i in range(len(intersections)):
                intersections[i].applyBC(roads, o, t)
            ### advance FV scheme
            for i in range(len(roads)):
                rhoL, rhoR = roads[i].rhoLR(o, limiter)
                if solver=='LxF':
                    # Lax Friedrichs
                    Frho = 0.5*(Flux(rhoL, roads[i].Umax, roads[i].rhomax) + Flux(rhoR, roads[i].Umax, roads[i].rhomax)) - 0.5*roads[i].dx/dt*(rhoR - rhoL)
                if solver=='upwind':
                    s = 1.-(rhoL+rhoR)
                    Frho = np.zeros_like(rhoL)
                    Frho[np.nonzero((s>0.)&(rhoL<.5))] = Flux(rhoL[np.nonzero((s>0.)&(rhoL<.5))], roads[i].Umax, roads[i].rhomax)
                    Frho[np.nonzero((s<0.)&(rhoR>.5))] = Flux(rhoR[np.nonzero((s<0.)&(rhoR>.5))], roads[i].Umax, roads[i].rhomax)
                    #Frho[np.nonzero((s>0.))] = Flux(rhoL[np.nonzero((s>0.))])
                    #Frho[np.nonzero((s<0.))] = Flux(rhoR[np.nonzero((s<0.))])
                    Frho[np.nonzero((rhoL>.5)&(.5>rhoR))] = Flux(.5, roads[i].Umax, roads[i].rhomax)

                ri = roads[i].roadIn
                ro = roads[i].roadOut
                if ri>-1:
                    Frho[0] = min(demand(rhoL[0], roads[ri].Umax, roads[ri].rhomax),
                                  supply(rhoR[0], roads[ i].Umax, roads[ri].rhomax))
                if ro>-1:
                    Frho[-1] = min(demand(rhoL[-1], roads[ i].Umax, roads[ i].rhomax),
                                   supply(rhoR[-1], roads[ro].Umax, roads[ro].rhomax))

                roads[i].rho[o:-o] -= dt/roads[i].dx*(Frho[1:] - Frho[0:-1])
        # end loop for order
        if o==2:
            for i in range(len(roads)):
                roads[i].rho = 0.5*(roads[i].rho + roads[i].rho_tmp)

    # end while
    return roads

def BC_inflow(rho,o,t):
    rho[0] = 0.5#max(0,np.sin(6*t))
    if o==2:
        rho[1] = cp(rho[0])
    return rho

def BC_outflow(rho,o,t):
    for i in range(1,o+1):
        rho[-o+i-1] = cp(rho[-o+i-2])
    return rho

#def BC_coupleGreen(rho,o,roadIn, roadOut):
#    for i in range(1,o+1):
#        rho[roadIn,-o+i-1] = cp(rho[roadOut,o+i-1])
#        rho[roadOut,o-i] = cp(rho[roadIn,-o-i])
#    return rho
#
#def BC_coupleRed(rho,o,roadIn, roadOut):
#    for i in range(1,o+1):
#        rho[roadIn,-o+i-1] = 1.
#        rho[roadOut,o-i] = 0.
#    return rho




def minmod(a,b):
    return 0.5*(np.sign(a)+np.sign(b))*np.minimum(np.abs(a),np.abs(b))

def maxmod(x,y):
    return 0.5*(np.sign(x)+np.sign(y))*np.maximum(np.abs(x),np.abs(y))

def superbee(x,y):
    return maxmod(minmod(x,2*y),minmod(2*x,y))

def mc(x,y,z):
    # elements with unequal signs
    dif = (((x<0.)&(y<0.)&(z<0.))!=True)&(((x>0.)&(y>0.)&(z>0.))!=True)
    x[np.nonzero(dif)]=0.0
    y[np.nonzero(dif)]=0.0
    z[np.nonzero(dif)]=0.0
    # all nonzero elements have the same sign now
    return np.sign(x)*np.minimum(np.abs(x),np.minimum(np.abs(y),np.abs(z)))

def main(n,o, limiter, solver):

    roads = [Road() for i in range(3)]

    roads[0].initialize(n,        np.array((-10.0,0.0)), np.array((0.0,0.0)), o, 1, 1, 0.)
    roads[1].initialize(int(n/2), np.array((0.0,0.0)), np.array((1.0,0.0)), o, .1, 1, 0.)
    roads[2].initialize(int(n/2), np.array((1.0,0.0)), np.array((10.0,0.0)), o, 1, 1, 0.)

    roads[0].roadIn=-1
    roads[0].roadOut=1
    roads[1].roadIn=0
    roads[1].roadOut=2
    roads[2].roadIn=1
    roads[2].roadOut=-1

    intersections = [Intersections() for i in range(4)]

    intersections[0].initialize(np.array(([-1])), np.array(([ 0])))
    intersections[1].initialize(np.array(([ 0])), np.array(([ 1])))
    intersections[2].initialize(np.array(([ 1])), np.array(([ 2])))
    intersections[3].initialize(np.array(([ 2])), np.array(([-1])))

    T = 50
    Tintervals = np.linspace(0.,1.*T,num=(T+1)*10)
    tshow=1
    for ti in range(Tintervals.shape[0]-1):
        if Tintervals[ti+2]>30:
            roads[1].Umax = 1.
        roads = ConsLaw(roads, intersections, Tintervals[ti:ti+2], limiter, solver, o)
        #print(roads[0].rho)
        #print(roads[1].rho)
        #print(roads[2].rho)
        #return
        if Tintervals[ti+2]>tshow:
            draw(roads, Tintervals[ti+2], o, True)
            tshow+=1



if __name__ == "__main__":

    from optparse import OptionParser
    usage = "usage: %prog [var=value]"
    p = OptionParser(usage)
    p.add_option("-d")
    p.add_option("--n", type="int", help="number of points")
    p.add_option("--order", type="int", help="order of the method")
    p.add_option("--method", type="string", help="which method")
    p.add_option("--limiter", type="string", help="which limiter")
    (opts, args) = p.parse_args()

    if opts.n == None:
        n = 250
    else:
        n = opts.n

    if opts.order == None:
        o = 2
    else:
        o = opts.order

    if opts.method == None:
        method = 'upwind'
    else:
        method = opts.method
        raise Exception("Not yet ready!")

    if opts.limiter == None:
        limiter = 'minmod'
    else:
        limiter = opts.limiter

    main(n,o,limiter,method)

