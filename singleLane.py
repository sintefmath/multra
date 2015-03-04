import numpy as np
import pylab as pl
from copy import copy as cp
import time

def main(n,o, limiter, solver):
    Length = 2.
    dx = Length/n
    x = dx/2. + np.arange(n)*dx
    rho = initializeRho(x,o)

    pl.ion()
    #pl.figure()
    line, = pl.plot(x,rho[o:-o])
    line.axes.set_ylim(-.1,1.1)
    pl.draw()

    #pl.plot(x,rho[o:-o],label='t=0')

    T = 10
    Tintervals = np.linspace(0.,1.*T,num=T+1)
    for ti in range(Tintervals.shape[0]-1):
        print Tintervals[ti:ti+2]
        rho = ConsLaw(x, rho, Tintervals[ti:ti+2], limiter, solver, o, line)
        #pl.plot(x,rho[o:-o],label='t=%0.2f'%Tintervals[ti+1])
    #pl.legend()
    #pl.show()

def ConsLaw(x, rho, Tinterval, limiter, solver, o, line):

    dx = x[1]-x[0]
    n = x.shape[0]

    t = cp(Tinterval[0])

    while t<Tinterval[1]:
        DF = Flux_drho(rho)
        eig = max(np.abs(DF))
        if (o==1):
            CFL = 0.99
        if (o==2):
            CFL = 0.49

        dt=1.*CFL*dx/eig

        if t+dt>Tinterval[1]:
            dt=Tinterval[1]-t
        t=t+dt

        if o==2:
            saverho = 1.*rho
        for internalsteps in range(1,o+1):
            rho = applyBoundaryConditions(rho,o,t)
            ## construct uL and uR
            if o==1:
                rhoL = rho[0:-1]
                rhoR = rho[1:]
            if o==2:
                # apply slope limiter
                a = rho[1:]-rho[0:-1]
                if limiter=='minmod':
                    S = minmod(a[0:-1],a[1:])
                if limiter=='superbee':
                    S = superbee(a[0:-1],a[1:])
                if limiter=='mc':
                    S = mc(2.*a[0:-1],.5*(rho[2:]-rho[:-2]),2.*a[1:])
                rhoL = rho[1:-2] + 0.5*S[0:-1]
                rhoR = rho[2:-1] - 0.5*S[1:]
            if solver=='LxF':
                # Lax Friedrichs
                Frho = 0.5*(Flux(rhoL) + Flux(rhoR)) - 0.5*dx/dt*(rhoR - rhoL)
            if solver=='upwind':
                s = 1.-(rhoL+rhoR)
                Frho = np.zeros_like(rhoL)
                Frho[np.nonzero((s>0.)&(rhoL<.5))] = Flux(rhoL[np.nonzero((s>0.)&(rhoL<.5))])
                Frho[np.nonzero((s<0.)&(rhoR>.5))] = Flux(rhoR[np.nonzero((s<0.)&(rhoR>.5))])
                #Frho[np.nonzero((s>0.))] = Flux(rhoL[np.nonzero((s>0.))])
                #Frho[np.nonzero((s<0.))] = Flux(rhoR[np.nonzero((s<0.))])
                Frho[np.nonzero((rhoL>.5)&(.5>rhoR))] = Flux(.5)
            rho[o:-o] -= dt/dx*(Frho[1:] - Frho[0:-1])
        # end loop for order
        if o==2:
            rho = 0.5*(rho + saverho)

        line.set_ydata(rho[o:-o])
        pl.draw()
        #time.sleep(0.05)

    # end while
    return rho


def Flux(rho):
    umax = 1.
    return rho*umax*(1.-rho)

def Flux_drho(rho):
    umax = 1.
    return umax*(1.-2*rho)

def initializeRho(x,o):
    # initialize function
    rho = 0.*x
  #  rho[np.nonzero((x>.75)&(x<1.25))]=1.
  #  x_ = x%1.
  #  rho = .5 + 0.*x_
  #  dx = x[1] - x[0]

  #  rho[np.nonzero((x<2.))]=1.

  #  rho[np.nonzero((x_>.25)&(x_<.75))]=1.
  #  # as this is a FV method take the integral in the cells containing a jump
  #  inde = np.nonzero((x_-.5*dx<.25)&(x_+.5*dx>.25))
  #  rho[inde]=(.25-(x_[inde]-.5*dx))/dx*.5 + (x_[inde]+.5*dx-.25)/dx*1.
  #  ### case one with a sin function ###
  #  rho[np.nonzero(x_>.5)] = np.sin(np.pi*1.5 + 6*np.pi*x_[np.nonzero(x_>.5)])/4. + .75
  #  ### case two with a jump ###
  #  #inde = np.nonzero((x_-.5*dx<.75)&(x_+.5*dx>.75))
  #  #rho[inde]=(.75-(x_[inde]-.5*dx))/dx*1. + (x_[inde]+.5*dx-.75)/dx*.5
  #  rho[x<1] = 0.5
  #  rho[x>2] = 0.5
  #  rho /= 1.

    # add ghost cells
    for i in range(1,o+1):
        rho = np.hstack([0.,rho,0.])

    return applyBoundaryConditions(rho,o,0)

def applyBoundaryConditions(rho,o,t):
    #rho[0] = rho[1]
    #rho[-1] = rho[-2]
    #return rho

    ### inflow ###
    #if (t>6):
    #    rho[0] = 0
    #else:
    #    rho[0] = max(0,np.sin(t))
    rho[0] = max(0,np.sin(6*t))
    if o==2:
        rho[1] = cp(rho[0])

    ### outflow ###
    if (t<6 or t>8):
        rho[-1] = 1.
        if o==2:
            rho[-2] = cp(rho[-1])
    else:
        rho[-1] = 0.
        if o==2:
            rho[-2] = cp(rho[-1])

    #for i in range(1,o+1):
    #    rho[o-i]=rho[o-i+1]
    #    rho[-o+i-1]=rho[-o+i-2]

    return rho

def minmod(a,b):
    return 0.5*(np.sign(a)+np.sign(b))*np.minimum(np.abs(a),np.abs(b))

def maxmod(x,y):
    return 0.5*(np.sign(x)+np.sign(y))*np.maximum(np.abs(x),np.abs(y))

def superbee(x,y):
    return maxmod(minmod(x,2*y),minmod(2*x,y))

def mc(x,y,z):
    # elements with not equal signs
    dif = (((x<0.)&(y<0.)&(z<0.))!=True)&(((x>0.)&(y>0.)&(z>0.))!=True)
    x[np.nonzero(dif)]=0.0
    y[np.nonzero(dif)]=0.0
    z[np.nonzero(dif)]=0.0
    # all nonzero elements have the same sign now
    return np.sign(x)*np.minimum(np.abs(x),np.minimum(np.abs(y),np.abs(z)))

if __name__ == "__main__":

    main(200,2,'minmod','upwind')

    #from optparse import OptionParser
    #usage = "usage: %prog [var=value]"
    #p = OptionParser(usage)
    #p.add_option("--n", type="int", help="number of points")
    #p.add_option("--order", type="int", help="first or second order method")
    #(opts, args) = p.parse_args()

    #if opts.n == None:
    #    n = 50
    #else:
    #    n = opts.n

    #if opts.order == None:
    #    o = 1
    #else:
    #    o = opts.order
    #    
    #main(n,o)


