#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import configurationPlotter as cfgPlt
from pylab import figure, axes, plot, xlabel, ylabel, title, grid, savefig, show, xlim, ylim
import matplotlib.pyplot as plt

# FEIR folders...
baseDirLocal   = 'C:/data/RegValidation/outFEIR/'

# fast masked...
cfgLstFastL_MuNeg8     = [ 'cfg200',
                           'cfg201',
                           'cfg202' ]

cfgLstFastL_MuNeg7     = [ 'cfg203',
                           'cfg204',
                           'cfg205' ]

cfgLstFastL_MuNeg6     = [ 'cfg206',
                           'cfg207',
                           'cfg208' ]

cfgLstFastL_MuNeg5     = [ 'cfg209',
                           'cfg210',
                           'cfg211' ]

# h masked...
cfgLstHL_MuNeg8     = [ 'cfg212',
                        'cfg213',
                        'cfg214' ]

cfgLstHL_MuNeg7     = [ 'cfg215',
                        'cfg216',
                        'cfg217' ]

cfgLstHL_MuNeg6     = [ 'cfg218',
                        'cfg219',
                        'cfg220' ]

cfgLstHL_MuNeg5     = [ 'cfg221',
                        'cfg222',
                        'cfg223' ]


plt_MuNeg8F = cfgPlt.configurationPlotter( baseDirLocal, cfgLstFastL_MuNeg8, 3 )
plt_MuNeg7F = cfgPlt.configurationPlotter( baseDirLocal, cfgLstFastL_MuNeg7, 3 )
plt_MuNeg6F = cfgPlt.configurationPlotter( baseDirLocal, cfgLstFastL_MuNeg6, 3 )
plt_MuNeg5F = cfgPlt.configurationPlotter( baseDirLocal, cfgLstFastL_MuNeg5, 3 )


for p in [ plt_MuNeg8F, plt_MuNeg7F, plt_MuNeg6F, plt_MuNeg5F ] :
    # breast
    p.breastFiles.insert( 1, 'n.a.' )
    p.breastInitialMean.insert( 1, p.breastInitialMean[0] )
    p.breastInitialPercentile.insert( 1, p.breastInitialPercentile[0] )
    # lesion
    p.lesionFiles.insert( 1, 'n.a.' )
    p.lesionInitialMean.insert( 1, p.breastInitialMean[0] )
    p.lesionInitialPercentile.insert( 1, p.breastInitialPercentile[0] )
    # xVals
    p.xVals.insert( 1, 0.0 )


# currently the old mu=0 values are not available... thus temporarily they are inserted manullay
plt_MuNeg8F.breastMeanVals.insert( 1, 1.095 ) # taken from excel file
plt_MuNeg8F.breastPercentileVals.insert( 1, 2.189 ) # taken from excel file
plt_MuNeg8F.lesionMeanVals.insert( 1, 1.356 ) # taken from excel file
plt_MuNeg8F.lesionPercentileVals.insert( 1, 1.704 ) # taken from excel file



plt_MuNeg7F.breastMeanVals.insert( 1, 1.083 ) # taken from excel file
plt_MuNeg7F.breastPercentileVals.insert( 1, 2.168 ) # taken from excel file
plt_MuNeg7F.lesionMeanVals.insert( 1, 1.218 ) # taken from excel file
plt_MuNeg7F.lesionPercentileVals.insert( 1, 1.469 ) # taken from excel file



plt_MuNeg6F.breastMeanVals.insert( 1, 1.110 ) # taken from excel file
plt_MuNeg6F.breastPercentileVals.insert( 1, 2.216 ) # taken from excel file
plt_MuNeg6F.lesionMeanVals.insert( 1, 1.187 ) # taken from excel file
plt_MuNeg6F.lesionPercentileVals.insert( 1, 1.406 ) # taken from excel file

plt_MuNeg5F.breastMeanVals.insert( 1, 1.167 ) # taken from excel file
plt_MuNeg5F.breastPercentileVals.insert( 1, 2.307 ) # taken from excel file
plt_MuNeg5F.lesionMeanVals.insert( 1, 1.246 ) # taken from excel file
plt_MuNeg5F.lesionPercentileVals.insert( 1, 1.462     ) # taken from excel file


# H results
plt_MuNeg8H = cfgPlt.configurationPlotter( baseDirLocal, cfgLstHL_MuNeg8, 3 )
plt_MuNeg7H = cfgPlt.configurationPlotter( baseDirLocal, cfgLstHL_MuNeg7, 3 )
plt_MuNeg6H = cfgPlt.configurationPlotter( baseDirLocal, cfgLstHL_MuNeg6, 3 )
plt_MuNeg5H = cfgPlt.configurationPlotter( baseDirLocal, cfgLstHL_MuNeg5, 3 )


for p in [ plt_MuNeg8H, plt_MuNeg7H, plt_MuNeg6H, plt_MuNeg5H ] :
    # breast
    p.breastFiles.insert( 1, 'n.a.' )
    p.breastInitialMean.insert( 1, p.breastInitialMean[0] )
    p.breastInitialPercentile.insert( 1, p.breastInitialPercentile[0] )
    # lesion
    p.lesionFiles.insert( 1, 'n.a.' )
    p.lesionInitialMean.insert( 1, p.breastInitialMean[0] )
    p.lesionInitialPercentile.insert( 1, p.breastInitialPercentile[0] )
    # xVals
    p.xVals.insert( 1, 0.0 )


# currently the old mu=0 values are not available... thus temporarily they are inserted manullay
plt_MuNeg8H.breastMeanVals.insert( 1, 0.844 ) # taken from excel file
plt_MuNeg8H.breastPercentileVals.insert( 1, 1.661 ) # taken from excel file
plt_MuNeg8H.lesionMeanVals.insert( 1, 1.085 ) # taken from excel file
plt_MuNeg8H.lesionPercentileVals.insert( 1, 1.493 ) # taken from excel file



plt_MuNeg7H.breastMeanVals.insert( 1, 0.871 ) # taken from excel file
plt_MuNeg7H.breastPercentileVals.insert( 1, 1.691 ) # taken from excel file
plt_MuNeg7H.lesionMeanVals.insert( 1, 0.928 ) # taken from excel file
plt_MuNeg7H.lesionPercentileVals.insert( 1, 1.207 ) # taken from excel file



plt_MuNeg6H.breastMeanVals.insert( 1, 0.903 ) # taken from excel file
plt_MuNeg6H.breastPercentileVals.insert( 1, 1.765 ) # taken from excel file
plt_MuNeg6H.lesionMeanVals.insert( 1, 0.874 ) # taken from excel file
plt_MuNeg6H.lesionPercentileVals.insert( 1, 1.084 ) # taken from excel file

plt_MuNeg5H.breastMeanVals.insert( 1, 0.972 ) # taken from excel file
plt_MuNeg5H.breastPercentileVals.insert( 1, 1.892 ) # taken from excel file
plt_MuNeg5H.lesionMeanVals.insert( 1, 0.917 ) # taken from excel file
plt_MuNeg5H.lesionPercentileVals.insert( 1, 1.101 ) # taken from excel file


