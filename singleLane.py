import numpy as np
import pylab as pl
from copy import copy as cp
import time

def main(n,o, limiter, solver):
    Length = 2.
    dx = Length/n
    x = dx/2. + np.arange(n)*dx
    rho_r1 = initializeRho(x,o)
    rho_r2 = initializeRho(x,o)
    rho = np.vstack([rho_r1,rho_r2])

    pl.ion()
    #pl.figure()
    line1, line2, = pl.plot(x,rho[0,o:-o],'g',Length+x,rho[1,o:-o],'r')
    line1.axes.set_ylim(-.1,1.1)
    line2.axes.set_ylim(-.1,1.1)
    pl.draw()

    #pl.plot(x,rho[o:-o],label='t=0')

    T = 20
    Tintervals = np.linspace(0.,1.*T,num=T+1)
    for ti in range(Tintervals.shape[0]-1):
        #print Tintervals[ti:ti+2]
        rho = ConsLaw(x, rho, Tintervals[ti:ti+2], limiter, solver, o, line1, line2)
        #pl.plot(x,rho[o:-o],label='t=%0.2f'%Tintervals[ti+1])
    #pl.legend()
    #pl.show()

def ConsLaw(x, rho, Tinterval, limiter, solver, o, line1, line2):

    dx = x[1]-x[0]
    n = x.shape[0]

    t = cp(Tinterval[0])

    count = 0
    while t<Tinterval[1]:
        count += 1

        DF = Flux_drho(rho[0,o:-o])
        eig = max(np.abs(DF))
        DF = Flux_drho(rho[1,o:-o])
        eig = max(eig,max(np.abs(DF)))

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
            rho[0,:] = BC_inflow(rho[0,:],o,t)
            if (t<6 or (t>8 and t<10) or (t>12 and t<15) or (t>16 and t<17) ):
                rho = BC_coupleRed(rho,o,0,1)
            else:
                rho = BC_coupleGreen(rho,o,0,1)
            rho[1,:] = BC_outflow(rho[1,:],o,t)
            ## construct uL and uR
            if o==1:
                rhoL = rho[:,0:-1]
                rhoR = rho[:,1:]
            if o==2:
                # apply slope limiter
                a = rho[:,1:]-rho[:,0:-1]
                if limiter=='minmod':
                    S = minmod(a[:,0:-1],a[:,1:])
                if limiter=='superbee':
                    S = superbee(a[:,0:-1],a[:,1:])
                if limiter=='mc':
                    S = mc(2.*a[:,0:-1],.5*(rho[:,2:]-rho[:,:-2]),2.*a[:,1:])
                rhoL = rho[:,1:-2] + 0.5*S[:,0:-1]
                rhoR = rho[:,2:-1] - 0.5*S[:,1:]
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
            rho[:,o:-o] -= dt/dx*(Frho[:,1:] - Frho[:,0:-1])
        # end loop for order
        if o==2:
            rho = 0.5*(rho + saverho)

        if (count%15==0):
            line1.set_ydata(rho[0,o:-o])
            line2.set_ydata(rho[1,o:-o])
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
    # add ghost cells
    for i in range(1,o+1):
        rho = np.hstack([0.,rho,0.])
    return rho

def BC_inflow(rho,o,t):
    ### inflow ###
    #if (t>6):
    #    rho[0] = 0
    #else:
    #    rho[0] = max(0,np.sin(t))
    rho[0] = max(0,np.sin(6*t))
    if o==2:
        rho[1] = cp(rho[0])
    return rho

def BC_outflow(rho,o,t):
    for i in range(1,o+1):
        rho[-o+i-1] = rho[-o+i-2]
    return rho

def BC_coupleGreen(rho,o,roadIn, roadOut):
    for i in range(1,o+1):
        rho[roadIn,-o+i-1] = cp(rho[roadOut,o+i-1])
        rho[roadOut,o-i] = cp(rho[roadIn,-o-i])
    return rho

def BC_coupleRed(rho,o,roadIn, roadOut):
    for i in range(1,o+1):
        rho[roadIn,-o+i-1] = 1.
        rho[roadOut,o-i] = 0.
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

    main(800,2,'minmod','upwind')

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


