import numpy
import matplotlib
matplotlib.use('Agg')
from scipy.cluster.vq import *
import pylab
pylab.close()

# generate some random xy points and
# give them some striation so there will be "real" groups.
xy = numpy.random.rand(30,2)
xy[3:8,1] -= .9
xy[22:28,1] += .9

# make some z vlues
z = numpy.sin(xy[:,1]-0.2*xy[:,1])

# whiten them
z = whiten(z)

# let scipy do its magic (k==3 groups)
res, idx = kmeans2(numpy.array(zip(xy[:,0],xy[:,1],z)),3)

# convert groups to rbg 3-tuples.
colors = ([([0,0,0],[1,0,0],[0,0,1])[i] for i in idx])

# show sizes and colors. each color belongs in diff cluster.
pylab.scatter(xy[:,0],xy[:,1],s=20*z+9, c=colors)
pylab.show()