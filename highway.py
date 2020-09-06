import numpy as np
from copy import copy as cp

from trafficPDE import *

import matplotlib.pyplot as pl
from openGLUtils import *

import time

def plot(roads, t, o, save):
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
        if Tintervals[ti+1]>30:
            roads[1].Umax = 1.
        roads = ConsLaw(roads, intersections, Tintervals[ti:ti+2], limiter, solver, o)
        #print(roads[0].rho)
        #print(roads[1].rho)
        #print(roads[2].rho)
        #return
        if Tintervals[ti+1]>tshow:
            #plot(roads, Tintervals[ti+1], o, False)
            draw(roads, Tintervals[ti+1], o, Tintervals)
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

    initGL()

    main(n,o,limiter,method)

