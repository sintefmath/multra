import linear_1d as linear_1d
import linear_2d as linear_2d
import linear_acoustics_1d as linear_acoustics_1d

import pylab as pl

def stillWorking():
    linear_1d.linear(1000, 1., 1)
    linear_1d.linear(1000, 1., 2, 'minmod')

    linear_acoustics_1d.linear(1000, 1., 1)
    linear_acoustics_1d.linear(1000, 1., 2, 'minmod')

    linear_2d.linear(200, 200, .5, 1)
    linear_2d.linear(200, 200, .5, 2, 'minmod')

    pl.show()
