#! /usr/bin/env python 
# -*- coding: utf-8 -*-



import modelDeformationVisualiser as visualiser
import meshStatistics as meshStat
import numpy as np
import xmlModelReader as xRead
from scipy.stats import scoreatpercentile
import matplotlib.pyplot as plt
import loadingFunction as lf
import modelDeformationVisualiser


class meshQualityTracker:
    ''' Track the tetrahedral element quality of the mesh over the iterations
    '''
    
    

    def __init__( self, xmlModelFileName='model.xml', deformationFileName='U.txt', keepStats=False, modelDeformVis=None ):
        ''' @param xmlModelFileName: file name of the xml-model (can be omitted of a visualiser was specified)
            @param deformationFileName: file name of the deformation file (can be omitted of a visualiser was specified)
            @param keepStats: Defines if the full (Ture) or only the percentile (Flase) statistics will be kept. Note: Huge memory requirements might arise! 
        '''
        if modelDeformVis == None :
            #
            # Read the model and the deformations
            #
            print('Reading model')
            reader = xRead.xmlModelReader( xmlModelFileName )
            print('Generating deformations')
            self.vis = visualiser.modelDeformationVisualiser( reader, deformationFileName )
        
        else:
            isinstance( modelDeformVis, modelDeformationVisualiser.modelDeformationVisualiser )
            self.vis = modelDeformVis
        
        self.qualityMeasures = ['RadiusRatio',
                        'MinAngle',
                        'EdgeRatio',
                        'Jacobian',
                        'ScaledJacobian',
                        'AspectBeta',
                        'AspectFrobenius',
                        'AspectGamma',
                        'AspectRatio',
                        'CollapseRatio',
                        'Condition',
                        'Distortion',
                        'RelativeSizeSquared',
                        'Shape',
                        'ShapeAndSize',
                        'Volume' ]
        self.percentileQuantities = [0,1,5,10,50,90,95,99,100]
        
        self._keepStats     = keepStats
        self._calculateStats( self._keepStats )
        
        


    def _calculateStats( self, keepStats ):
        
        print( 'Found %i deformed model versions.' % len( self.vis.deformedNodes ) )
        
        self.percentiles = {}
        self.stats       = []

        #
        # Prepare the data structures which hold the percentiles
        #        
        for qm in self.qualityMeasures:
            self.percentiles[ qm ] = {}
            
            for p in self.percentileQuantities :
                self.percentiles[qm][p] = []
        
        #
        # Evaluate each deformed model
        #
        for i in range( len( self.vis.deformedNodes ) ) :
            
            if np.mod( i, 10 ) == 0:
                print( 'Calculating statistics: %5i' % i )
            
            # Calculate the statistics for the current set of deformed nodes
            stats = meshStat.meshStatistics( self.vis.deformedNodes[i], self.vis.mldElements ) 
            
            # Keep the full statistics
            # NOTE: Huge memory requirements likely
            if keepStats :
                self.stats.append(stats)
            
            # For each quality measure evaluate the percentiles specified
            for qm in self.qualityMeasures :
                
                for p in self.percentileQuantities :
                    
                    self.percentiles[ qm ][ p ].append( scoreatpercentile( stats.qualityMeasures[ qm ], p ) )
            
        #
        # Convert the percentiles into numpy arrays
        #
        
        for qm in self.qualityMeasures:
            for p in self.percentileQuantities :
                self.percentiles[qm][p] = np.array( self.percentiles[qm][p] )

    
    
    
    def plotQualityMeasure( self, plotDir, loadShape, totalTime, qualityMeasure ):
        
        load, time = lf.loadingFunction(totalTime, loadShape, len( self.vis.deformedNodes ) )
        
        c000 = '#BF7130'
        c001 = '#FF9640'
        c005 = '#FFA200'
        c010 = '#FF7400'
        c050 = '#FF0000'
        c090 = '#104BA9'
        c095 = '#009999'
        c099 = '#447BD4'
        c100 = '#1D7373'
        
        l000 = '$p_{0\%}$'
        l001 = '$p_{1\%}$'
        l005 = '$p_{5\%}$'
        l010 = '$p_{10\%}$'
        l050 = '$p_{50\%}$'
        l090 = '$p_{90\%}$'
        l095 = '$p_{95\%}$'
        l099 = '$p_{99\%}$'
        l100 = '$p_{100\%}$'
    
        plt.rc( 'text', usetex=True )
        plt.rcParams['font.size']=16
        
        fig = plt.figure()
        plt.hold( True )
        ax1 = fig.gca()
        ax1.plot( time, self.percentiles[qualityMeasure ][  0], c000, label = l000 )
        ax1.plot( time, self.percentiles[qualityMeasure ][  1], c001, label = l001 )
        ax1.plot( time, self.percentiles[qualityMeasure ][  5], c005, label = l005 )
        ax1.plot( time, self.percentiles[qualityMeasure ][ 10], c010, label = l010 )
        ax1.plot( time, self.percentiles[qualityMeasure ][ 50], c050, label = l050 )
        ax1.plot( time, self.percentiles[qualityMeasure ][ 90], c090, label = l090 )
        ax1.plot( time, self.percentiles[qualityMeasure ][ 95], c095, label = l095 )
        ax1.plot( time, self.percentiles[qualityMeasure ][ 90], c099, label = l099 )
        ax1.plot( time, self.percentiles[qualityMeasure ][100], c100, label = l100 )
        ax1.set_xlabel( '$t\;\mathrm{[s]}$' )
        ax1.set_ylabel( '$q_\mathrm{' + qualityMeasure + '}$' )
        ax1.grid( color = 'gray', linestyle='-' )
        
        ax2 = ax1.twinx()
        ax2.plot( time, load, 'r-', label = '$L_\mathrm{rel}$' )
        ax2.set_ylabel('$L_\mathrm{rel}$')
        ax2.set_ylim( bottom=0, top=1.1 )
        
        plt.hold( False )
        
        fig.show()
        fig.savefig( plotDir + qualityMeasure + '.pdf' )
        fig.savefig( plotDir + qualityMeasure + '.png', dpi = 300 )




if __name__ == '__main__':
    
    simDir         = 'W:/philipsBreastProneSupine/referenceState/00_float_double/float/'
    modelFileName  = 'modelFat_prone1G_it050000_totalTime05_rampflat4.xml'
    deformFileName = 'U_modelFat_prone1G_it050000_totalTime05_rampflat4.txt'
    
    mqt = meshQualityTracker( simDir + modelFileName, simDir + deformFileName, 'RadiusRatio', keepStats=True )
    
    #
    # evaluate the travel distance
    #
    r=range(250, 499)
    travelDist = np.zeros( mqt.vis.displacements[0].shape[0] )
    
    for i in r:
        d =  mqt.vis.displacements[i] - mqt.vis.displacements[i-1]
        d = np.sqrt( d[:,0] * d[:,0] + 
                     d[:,1] * d[:,1] + 
                     d[:,2] * d[:,2] )  
        
        travelDist = travelDist + d
    
    #
    # The node which moves the most.... 
    #
    a = np.nonzero( np.max(travelDist) == travelDist )

    #
    # Now needs to be related to the elements:
    #  find the elements which contain this node...
    e = np.nonzero( mqt.vis.mldElements == a )




