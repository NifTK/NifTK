#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import xmlModelParameterSweeper as xSweep

numItMin =  10000
numItMax = 400000
itInc    =   5000

referenceModelFile = 'Q:/philipsBreastProneSupine/referenceState/00_load_varIT/referenceModel/model_prone1G_totalTime05_rampflat4.xml'
outDir             = 'Q:/philipsBreastProneSupine/referenceState/00_load_varIT/'

iterations = range( numItMin, numItMax + 1, itInc )

totalSimTime = 5.0
timeStepID   = 'TimeStep'

ids       = []
timeSteps = []

for it in iterations:
    ids.append( str('_it%07i' % it) )
    timeSteps.append( totalSimTime / it)


xSweep.xmlModelParameterSweeper( referenceModelFile, timeStepID, timeSteps, ids, outDir )