import linear_1d as linear_1d
import linear_2d as linear_2d
import linear_acoustics_1d as linear_acoustics_1d
import linear_acoustics_2d as linear_acoustics_2d
import induction_equation_2d as induction_equation_2d

import pylab as pl

def stillWorking():
    pl.figure(1)
    linear_1d.linear(1000, 1., 1)
    linear_1d.linear(1000, 1., 2, 'minmod')

    pl.figure(2)
    linear_2d.linear(100, 105, .5, 1)
    pl.colorbar()
    pl.figure(3)
    linear_2d.linear(100, 105, .5, 2, 'minmod')
    pl.colorbar()

    pl.figure(4)
    linear_acoustics_1d.linear(1000, 1.5, 1)
    linear_acoustics_1d.linear(1000, 1.5, 2, 'minmod')

    pl.figure(5)
    linear_acoustics_2d.linear(100, 105, 1.1, 1)
    pl.colorbar()
    pl.figure(6)
    linear_acoustics_2d.linear(100, 105, 1.1, 2, 'minmod')
    pl.colorbar()

    pl.figure(5)
    linear_acoustics_2d.linear(100, 105, 1.1, 1)
    pl.colorbar()
    pl.figure(6)
    linear_acoustics_2d.linear(100, 105, 1.1, 2, 'minmod')
    pl.colorbar()

    pl.figure(5)
    induction_equation_2d.linear(100, 105)
    pl.colorbar()
    pl.figure(6)
    induction_equation_2d.linear(100, 105)
    pl.colorbar()

    pl.show()
