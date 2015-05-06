import linear_1d as linear_1d
import linear_2d as linear_2d
import linear_acoustics_1d as linear_acoustics_1d
import linear_acoustics_2d as linear_acoustics_2d

import pylab as pl

def stillWorking():
    pl.figure(1)
    linear_1d.linear(1000, 1., 1)
    linear_1d.linear(1000, 1., 2, 'minmod')

    pl.figure(2)
    linear_2d.linear(200, 205, .5, 1)
    linear_2d.linear(200, 205, .5, 2, 'minmod')

    pl.figure(3)
    linear_acoustics_1d.linear(1000, .5, 1)
    linear_acoustics_1d.linear(1000, .5, 2, 'minmod')

    pl.figure(4)
    linear_acoustics_2d.linear(200, 205, .1, 1)
    linear_acoustics_2d.linear(200, 205, .1, 2, 'minmod')

    pl.show()
