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
    
    

    def __init__( self, xmlModelFileName='model.xml', deformationFileName='U.txt', qualityMeasure='RadiusRatio', keepStats=True, modelDeformVis=None ):
        ''' @param xmlModelFileName: file name of the xml-model (can be omitted of a visualiser was specified)
            @param deformationFileName: file name of the deformation file (can be omitted of a visualiser was specified)
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
        
        self.qualityMeasure = qualityMeasure
        self._keepStats     = keepStats
        self._calculateStats(self._keepStats, qualityMeasure)
        
        


    def _calculateStats( self, keepStats, qualityMeasure ):
        
        self.min           = []
        self.max           = []
        self.percentile01  = []
        self.percentile05  = []
        self.percentile10  = []
        self.percentile50  = []
        self.percentile90  = []
        self.percentile95  = []
        self.percentile99  = []
        
        print( 'Found %i deformed model versions.' % len( self.vis.deformedNodes ) )
        if not keepStats :
            for i in range( len( self.vis.deformedNodes ) ) :
                if np.mod( i,10 ) == 0:
                    print( 'Calculating statistics: %5i' % i )
                
                stats = meshStat.meshStatistics( self.vis.deformedNodes[i], self.vis.mldElements ) 
                
                self.min.append( scoreatpercentile( stats.qualityMeasures[ qualityMeasure ],   0.0 ) )
                self.max.append( scoreatpercentile( stats.qualityMeasures[ qualityMeasure ], 100.0 ) )
    
                self.percentile01.append( scoreatpercentile( stats.qualityMeasures[ qualityMeasure ],  1.0 ) )
                self.percentile05.append( scoreatpercentile( stats.qualityMeasures[ qualityMeasure ],  5.0 ) )
                self.percentile10.append( scoreatpercentile( stats.qualityMeasures[ qualityMeasure ], 10.0 ) )
                self.percentile50.append( scoreatpercentile( stats.qualityMeasures[ qualityMeasure ], 50.0 ) )
                self.percentile90.append( scoreatpercentile( stats.qualityMeasures[ qualityMeasure ], 90.0 ) )
                self.percentile95.append( scoreatpercentile( stats.qualityMeasures[ qualityMeasure ], 95.0 ) )
                self.percentile99.append( scoreatpercentile( stats.qualityMeasures[ qualityMeasure ], 99.0 ) )
        else:
            self.stats = []
           
            for i in range( len( self.vis.deformedNodes ) ) :
                if np.mod( i,10 ) == 0:
                    print( 'Calculating statistics: %5i' % i )
                
                self.stats.append( meshStat.meshStatistics( self.vis.deformedNodes[i], self.vis.mldElements ) ) 
                
                self.min.append( scoreatpercentile( self.stats[-1].qualityMeasures[ qualityMeasure ],   0.0 ) )
                self.max.append( scoreatpercentile( self.stats[-1].qualityMeasures[ qualityMeasure ], 100.0 ) )
    
                self.percentile01.append( scoreatpercentile( self.stats[-1].qualityMeasures[ qualityMeasure ],  1.0 ) )
                self.percentile05.append( scoreatpercentile( self.stats[-1].qualityMeasures[ qualityMeasure ],  5.0 ) )
                self.percentile10.append( scoreatpercentile( self.stats[-1].qualityMeasures[ qualityMeasure ], 10.0 ) )
                self.percentile50.append( scoreatpercentile( self.stats[-1].qualityMeasures[ qualityMeasure ], 50.0 ) )
                self.percentile90.append( scoreatpercentile( self.stats[-1].qualityMeasures[ qualityMeasure ], 90.0 ) )
                self.percentile95.append( scoreatpercentile( self.stats[-1].qualityMeasures[ qualityMeasure ], 95.0 ) )
                self.percentile99.append( scoreatpercentile( self.stats[-1].qualityMeasures[ qualityMeasure ], 99.0 ) ) 
            
        self.min = np.array( self.min )
        self.max = np.array( self.max )

        self.percentile01 = np.array( self.percentile01 )
        self.percentile05 = np.array( self.percentile05 )
        self.percentile10 = np.array( self.percentile10 )
        self.percentile50 = np.array( self.percentile50 )
        self.percentile90 = np.array( self.percentile90 )
        self.percentile95 = np.array( self.percentile95 )
        self.percentile99 = np.array( self.percentile99 )



        
    def getQualityMeasureResults( self, qualityMeasure ):
        self.qualityMeasure = qualityMeasure
        
        #
        # Need to recalculate as this was not saved locally
        #
        if not self._keepStats:
            self._calculateStats( self._keepStats, qualityMeasure )
            return
        
        #
        # Otherwise just go through the saved measurements
        #
        else:   
            self.min = []
            self.max = []
            self.percentile01  = []
            self.percentile05  = []
            self.percentile10  = []
            self.percentile50  = []
            self.percentile90  = []
            self.percentile95  = []
            self.percentile99  = []
            
            for i in range( len( self.vis.deformedNodes ) ) :
                
                self.min.append( scoreatpercentile( self.stats[i].qualityMeasures[ qualityMeasure ],   0.0 ) )
                self.max.append( scoreatpercentile( self.stats[i].qualityMeasures[ qualityMeasure ], 100.0 ) )
    
                self.percentile01.append( scoreatpercentile( self.stats[i].qualityMeasures[ qualityMeasure ],  1.0 ) )
                self.percentile05.append( scoreatpercentile( self.stats[i].qualityMeasures[ qualityMeasure ],  5.0 ) )
                self.percentile10.append( scoreatpercentile( self.stats[i].qualityMeasures[ qualityMeasure ], 10.0 ) )
                self.percentile50.append( scoreatpercentile( self.stats[i].qualityMeasures[ qualityMeasure ], 50.0 ) )
                self.percentile90.append( scoreatpercentile( self.stats[i].qualityMeasures[ qualityMeasure ], 90.0 ) )
                self.percentile95.append( scoreatpercentile( self.stats[i].qualityMeasures[ qualityMeasure ], 95.0 ) )
                self.percentile99.append( scoreatpercentile( self.stats[i].qualityMeasures[ qualityMeasure ], 99.0 ) ) 
            
            self.min = np.array( self.min )
            self.max = np.array( self.max )
    
            self.percentile01 = np.array( self.percentile01 )
            self.percentile05 = np.array( self.percentile05 )
            self.percentile10 = np.array( self.percentile10 )
            self.percentile50 = np.array( self.percentile50 )
            self.percentile90 = np.array( self.percentile90 )
            self.percentile95 = np.array( self.percentile95 )
            self.percentile99 = np.array( self.percentile99 )

    
    
    
    def plotQualityMeasure( self, plotDir, loadShape, totalTime ):
        
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
        ax1.plot( time, self.min,          c000, label = l000 )
        ax1.plot( time, self.percentile01, c001, label = l001 )
        ax1.plot( time, self.percentile05, c005, label = l005 )
        ax1.plot( time, self.percentile10, c010, label = l010 )
        ax1.plot( time, self.percentile50, c050, label = l050 )
        ax1.plot( time, self.percentile90, c090, label = l090 )
        ax1.plot( time, self.percentile95, c095, label = l095 )
        ax1.plot( time, self.percentile99, c099, label = l099 )
        ax1.plot( time, self.max,          c100, label = l100 )
        ax1.set_xlabel( '$t\;\mathrm{[s]}$' )
        ax1.set_ylabel( '$q_\mathrm{' + self.qualityMeasure + '}$' )
        ax1.grid( color = 'gray', linestyle='-' )
        
        ax2 = ax1.twinx()
        ax2.plot( time, load, 'r-', label = '$L_\mathrm{rel}$' )
        ax2.set_ylabel('$L_\mathrm{rel}$')
        ax2.set_ylim( bottom=0, top=1.1 )
        
        plt.hold( False )
        
        fig.show()
        fig.savefig( plotDir + self.qualityMeasure + '.pdf' )
        fig.savefig( plotDir + self.qualityMeasure + '.png', dpi = 300 )




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




