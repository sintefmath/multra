import numpy as np
from copy import copy as cp

from trafficPDE import *

import time

from openGLUtils import *

def main(n,o, limiter, solver):

    roads = [Road() for i in range(4)]
    #for i in range(len(roads)):
    #    roads[i].initializeRoad(n, np.array((-2.+i*2,.0)), np.array((-0.+i*2,0.)), o)

    roads[0].initialize(n,        np.array((-2.0,0.0)), np.array((-1.0,0.0)), o)
    roads[1].initialize(int(n/2), np.array((-1.0,0.0)), np.array(( -1.0,1.0)), o)
    roads[2].initialize(int(n/2), np.array(( -1,1.0)), np.array(( -1+0.4,1.0)), o)
    roads[3].initialize(int(n/2), np.array(( 0.4-1,1.0)), np.array(( 0.4-1,0.0)), o)

    roads[0].roadIn=-1
    roads[0].roadOut=1
    roads[1].roadIn=0
    roads[1].roadOut=2
    roads[2].roadIn=1
    roads[2].roadOut=3
    roads[3].roadIn=2
    roads[3].roadOut=-1
    
    intersections = [Intersections() for i in range(5)]

    intersections[0].initialize(np.array(([-1])), np.array(([ 0])))
    intersections[1].initialize(np.array(([ 0])), np.array(([ 1])), np.mat(([1])), np.array((0.,6., 10.,12., 15.,16., 17.,20.)) )
    intersections[2].initialize(np.array(([ 1])), np.array(([ 2])), np.mat(([1])), np.array((6.,10., 12.,15., 16.,17., 19.,20.)) )
    intersections[3].initialize(np.array(([ 2])), np.array(([ 3])), np.mat(([1])), np.array((8.,10., 12.,15., 16.,17., 19.,20.)) )
    intersections[4].initialize(np.array(([ 3])), np.array(([-1])))

    T = 30
    Tintervals = np.linspace(0.,1.*T,num=(T+1)*10)
    for ti in range(Tintervals.shape[0]-1):
        roads = ConsLaw(roads, intersections, Tintervals[ti:ti+2], limiter, solver, o)
        draw(roads, Tintervals[ti+1], o, Tintervals)


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

    if opts.limiter == None:
        limiter = 'minmod'
    else:
        limiter = opts.limiter

    initGL()

    main(n,o,limiter,method)

