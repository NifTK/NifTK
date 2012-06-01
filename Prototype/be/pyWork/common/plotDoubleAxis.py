#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt


def plotDoubleYAxis( xVals, y1Vals, y2Vals, xLabel, xLabelUnit, y1Label, y1LabelUnit, y2Label, y2LabelUnit, plotDirAndBaseName, printLegend=True, y1Max = None ):
    ''' Generate a twin-y-axis plot and save it to the given directory (.pdf and .png will be appended automatically)
    '''
    
    plt.rc( 'text', usetex=True )
    plt.rcParams['font.size']=16
    
    fig = plt.figure()
    plt.hold( True )
    ax1 = fig.gca()
    ax1.plot( xVals, y1Vals, 'b-', label = y1Label )
    ax2 = ax1.twinx()
    ax2.plot( xVals, y2Vals, 'r-', label = y2Label )
    ax1.set_xlabel( xLabelUnit        )
    ax1.set_ylabel( y1LabelUnit )
    ax2.set_ylabel( y2LabelUnit )
    ax1.set_ylim( bottom=0 )
    ax2.set_ylim( bottom=0, top=1.1 )
    ax1.grid( color = 'gray', linestyle='-' )
    
    if y1Max != None:
        ax1.set_ylim( top=y1Max )
    
    if printLegend:
        plt.legend( (ax1.get_lines(), ax2.get_lines()), (y1Label, y2Label), loc = 'lower right' )
    
    plt.hold( False )
    fig.show()
    fig.savefig( plotDirAndBaseName + '.pdf' )
    fig.savefig( plotDirAndBaseName + '.png', dpi = 300 )