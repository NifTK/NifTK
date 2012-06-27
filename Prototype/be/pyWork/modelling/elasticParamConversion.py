import numpy as np
import math


YoungsModulus = lambda lameMu, lameLambda : ( lameMu * (2 * lameMu + 3 * lameLambda) ) / (lameMu + lameLambda)

PoissonsRatio = lambda lameMu, lameLambda : lameLambda / ( 2 * (lameMu + lameLambda) )

lameLambda = lambda YoungsModulus, PoissonsRatio : YoungsModulus * PoissonsRatio / ( (1 + PoissonsRatio) * (1 - 2*PoissonsRatio) )

lameMu = lambda YoungsModulus, PoissonsRatio : YoungsModulus / (2 * (1 + PoissonsRatio) )

# TODO: Bulk and shear modulus
