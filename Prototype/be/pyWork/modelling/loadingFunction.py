#! /usr/bin/env python 
# -*- coding: utf-8 -*-



import numpy as np



def _calcPoly345RampN(time, N) :
            
    x = N*time.copy(); 
    load = 6.0 * x**5 - 15.0 * x**4 + 10.0 * x**3
    del x
    return load




def loadingFunction( tTotal, loadShape, numIterations ):
    
    # generate time and loading function
    
    time = np.arange( tTotal/numIterations, tTotal + 1e-10, tTotal/numIterations )
    load = time.copy() / tTotal                           # normalised time...
    
    #
    # N = 1.0 defaults to RAMP
    #
    N = 1.0

    if loadShape == 'RAMPFLAT':
        N = 2.0  # 1 part ramp up, 1 part keep constant...
        load = load * N     
    
    if loadShape == 'RAMPFLAT2':
        N = 3.0  # 1 part ramp up, 2 parts keep constant...
        load = load * N     
    
    if loadShape == 'RAMPFLAT4':
        N = 5.0  # 1 part ramp up, 4 parts keep constant...
        load = load * N     
    
    if loadShape == 'RAMPFLAT8':
        N = 9.0  # 1 part ramp up, 4 parts keep constant...
        load = load * N    
    
    #
    # poly345 shapes
    #
    if loadShape == 'POLY345':
        N = 1.0
        load = _calcPoly345RampN(load, N)
    
    if loadShape == 'POLY345FLAT':
        N = 2.0
        load = _calcPoly345RampN(load, N)
    
    if loadShape == 'POLY345FLAT2':
        N = 3.0
        load = _calcPoly345RampN(load, N)
    
    if loadShape == 'POLY345FLAT4':
        N = 5.0
        load = _calcPoly345RampN(load, N)

    if loadShape == 'POLY345FLAT8':
        N = 9.0
        load = _calcPoly345RampN(load, N)
    
    if loadShape == 'STEP':
        load = np.ones_like( time )
    
    #
    # Take care of "holding" part of loading function
    #
    load[time >  tTotal/N ] = 1.0
    
    return load, time 





if __name__ == '__main__':
    
    l,t = loadingFunction(10., 'POLY345', 500)
    pass
