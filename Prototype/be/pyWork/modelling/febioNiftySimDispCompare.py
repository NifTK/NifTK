#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import numpy as np
import os
import xmlModelReader as xRead
import modelDeformationVisualiser as mdVis
import mayaviPlottingWrap as mpw
import matplotlib.pyplot as plt
from mayavi import mlab

class febioNiftySimComparison:
    
    
    
    def __init__( self, niftySimModelName, niftySimDispFileName, febioASCIISceneName ) :
        
        self.niftySimModelName    = niftySimModelName
        self.niftySimDispFileName = niftySimDispFileName
        self.febioASCIISceneName  = febioASCIISceneName
        
        self._readFEBioDisplacedNodes()
        self._readNiftySimResults()
        self._calcMaxAndMeanDisplacements()
        
        pass
    
    
    
    
    def _readFEBioDisplacedNodes( self ):
        ''' Only the nodes are read
        '''
        
        f       = file(self.febioASCIISceneName, 'r')
        lines   = f.readlines()
        numCols = len( lines[2].split() )
        
        numRows = len( lines )        
        results = []        
        
        for i in range( 2, len( lines ) ) :
            
            # reached end of node positions?
            if lines[i].startswith('*'):
                break
            
            line = lines[i].split()
            
            # remove separating comma
            line[1]= float( line[1].replace(',', '') )
            line[2]= float( line[2].replace(',', '') )
            line[3]= float( line[3].replace(',', '') )
            results.append( np.array((line[1],line[2],line[3])) )
        
        self.febioDisplacedNodes = np.array( results )
        print( 'FEBio Reading done' )
       
       
       
        
    def _readNiftySimResults( self ):
        
        self.xmlReader = xRead.xmlModelReader( self.niftySimModelName )
        self.vis       =  mdVis.modelDeformationVisualiser( self.xmlReader, self.niftySimDispFileName )
        
    


    def plot3DResults( self ):
        
        scaleF = 1000
        mpw.plotArrayAs3DPoints( self.vis.deformedNodes[-1] * scaleF, (1, 0, 0) )
        mpw.plotArrayAs3DPoints( self.febioDisplacedNodes   * scaleF, (0, 1, 0) )

        mpw.plotVectorsAtPoints( scaleF * (self.febioDisplacedNodes - self.vis.deformedNodes[-1]) , scaleF * self.vis.deformedNodes[-1] )
        
        
        

    def plotNodalDisplacementHistogram( self, plotDir = None, xMax=None ):
        
        plt.rc( 'text', usetex=True )
        plt.rcParams['font.size']=16
        
        # calculate displacement vector length 
        # NOTE: This is given in mm
        self.absDisps = self.febioDisplacedNodes - self.vis.deformedNodes[-1]
        self.absDisps = np.sqrt( self.absDisps[:,0]**2 + self.absDisps[:,1]**2 + self.absDisps[:,2]**2 ) * 1000.
        
        # calculate the mean difference of the displacements        
        self.meanDispDif = np.mean( self.absDisps )
        
        fig = plt.figure()
        if xMax == None:
            plt.hist( self.absDisps, 150 ) 

        else:
            plt.hist( self.absDisps, bins=150, range = ( 0, xMax ) ) 

        plt.grid()
        ax  = fig.gca()
        ax.set_xlabel( '$\|u_\mathrm{niftySim} - u_\mathrm{FEBio} \|\;\mathrm{[mm]}$' )
        ax.set_ylabel( '$N$' )
        
        if xMax != None:
            ax.set_xlim( right=xMax )
            
        plt.show()
        
        if plotDir != None:
            
            # construct plotting file name
            plotName = plotDir + os.path.splitext( os.path.basename( self.niftySimModelName ) )[0] + '__vs__' + os.path.splitext( os.path.basename( self.febioASCIISceneName ) )[0]
            
            fig.savefig( plotName + '.pdf' )
            fig.savefig( plotName + '.png', dpi = 300 )
    
    
    
    
    def _calcMaxAndMeanDisplacements( self ):
        
        dispFEBio    = self.vis.mdlNodes - self.febioDisplacedNodes
        dispNiftySim = self.vis.displacements[-1]
        
        dispFEBio    = np.sqrt( dispFEBio[:,0]    ** 2 + dispFEBio[:,1]    ** 2 + dispFEBio[:,2]    ** 2 ) 
        dispNiftySim = np.sqrt( dispNiftySim[:,0] ** 2 + dispNiftySim[:,1] ** 2 + dispNiftySim[:,2] ** 2 ) 
        
        self.maxDispFEBio    = np.max( dispFEBio    )
        self.maxDispNiftySim = np.max( dispNiftySim )
        
        self.meanDispFEbio    = np.mean( dispFEBio    )
        self.meanDispNiftySim = np.mean( dispNiftySim )
        
        
        
        
    def setDefaultScene(self):
        f = mlab.gcf()
        f.scene.background =( 1.0, 1.0, 1.0 )
        
        #roll = 57.67368291090266
        #view = (-118.93865721738072, 64.957822912155606, 98.369333568183777, np.array([  7.49999809,  15.83817911,  11.66932011]))
        
        # look at the model from the side with white background
        view = (180.0, 90.0, 85.487641791319476, np.array([  7.49999809,  15.83817911,  11.66932011]))
        roll = 90.0
        
        mlab.view( *view )
        mlab.roll( roll )
        



if __name__ == '__main__' :
    
    #
    # H8
    #
    
    plotDir = 'W:/philipsBreastProneSupine/referenceState/boxModel/summary/hists/' 
    
#    # H8 disp
#    niftysimModel        = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001/box_H8_disp.xml'
#    niftySimDisplacement = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001/U_box_H8_disp.txt'
#    febioASCIIFile       = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/FEBio/disp_15_out.txt'
#    
#    comp_H8_disp = febioNiftySimComparison( niftysimModel, niftySimDisplacement, febioASCIIFile )
#    comp_H8_disp.plotNodalDisplacementHistogram( plotDir )
#    plt.close('all')
#    
#    # H8 force
#    niftysimModel        = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001/box_H8_force.xml'
#    niftySimDisplacement = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001/U_box_H8_force.txt'
#    febioASCIIFile       = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/FEBio/force_15_out.txt'
#    
#    comp_H8_force = febioNiftySimComparison( niftysimModel, niftySimDisplacement, febioASCIIFile )
#    comp_H8_force.plotNodalDisplacementHistogram( plotDir )
#    plt.close('all')
#    
#    # H8 grav
#    niftysimModel        = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001/box_H8_grav.xml'
#    niftySimDisplacement = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001/U_box_H8_grav.txt'
#    febioASCIIFile       = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/FEBio/grav_15_out.txt'
#    
#    comp_H8_grav = febioNiftySimComparison( niftysimModel, niftySimDisplacement, febioASCIIFile )
#    comp_H8_grav.plotNodalDisplacementHistogram( plotDir )
#    plt.close('all')
#    
#    
#    #
#    # T4
#    #
#    
#    # T4 disp
#    niftysimModel        = 'W:/philipsBreastProneSupine/referenceState/boxModel/T4/NH_D001/box_T4_disp.xml'
#    niftySimDisplacement = 'W:/philipsBreastProneSupine/referenceState/boxModel/T4/NH_D001/U_box_T4_disp.txt'
#    febioASCIIFile       = 'W:/philipsBreastProneSupine/referenceState/boxModel/T4/FEBio/disp_15_out.txt'
#    
#    comp_T4_disp = febioNiftySimComparison( niftysimModel, niftySimDisplacement, febioASCIIFile )
#    comp_T4_disp.plotNodalDisplacementHistogram( plotDir )
#    plt.close('all')
#    
#    # T4 force
#    niftysimModel        = 'W:/philipsBreastProneSupine/referenceState/boxModel/T4/NH_D001/box_T4_force.xml'
#    niftySimDisplacement = 'W:/philipsBreastProneSupine/referenceState/boxModel/T4/NH_D001/U_box_T4_force.txt'
#    febioASCIIFile       = 'W:/philipsBreastProneSupine/referenceState/boxModel/T4/FEBio/force_15_out.txt'
#    
#    comp_T4_force = febioNiftySimComparison( niftysimModel, niftySimDisplacement, febioASCIIFile )
#    comp_T4_force.plotNodalDisplacementHistogram( plotDir )
#    plt.close('all')
#    
#    # T4 grav
#    niftysimModel        = 'W:/philipsBreastProneSupine/referenceState/boxModel/T4/NH_D001/box_T4_grav.xml'
#    niftySimDisplacement = 'W:/philipsBreastProneSupine/referenceState/boxModel/T4/NH_D001/U_box_T4_grav.txt'
#    febioASCIIFile       = 'W:/philipsBreastProneSupine/referenceState/boxModel/T4/FEBio/grav_15_out.txt'
#    
#    comp_T4_grav = febioNiftySimComparison( niftysimModel, niftySimDisplacement, febioASCIIFile )
#    comp_T4_grav.plotNodalDisplacementHistogram( plotDir )
#    plt.close('all')
#    
#    
#    comp_T4_grav.plot3DResults()
#
#    
#    
#    #
#    # T4ANP
#    #
#    
#    # T4ANP disp
#    niftysimModel        = 'W:/philipsBreastProneSupine/referenceState/boxModel/T4ANP/NH_D001/box_T4ANP_disp.xml'
#    niftySimDisplacement = 'W:/philipsBreastProneSupine/referenceState/boxModel/T4ANP/NH_D001/U_box_T4ANP_disp.txt'
#    febioASCIIFile       = 'W:/philipsBreastProneSupine/referenceState/boxModel/T4/FEBio/disp_15_out.txt'
#    
#    comp_T4ANP_disp = febioNiftySimComparison( niftysimModel, niftySimDisplacement, febioASCIIFile )
#    comp_T4ANP_disp.plotNodalDisplacementHistogram( plotDir )
#    plt.close('all')
#    
#    # T4ANP force
#    niftysimModel        = 'W:/philipsBreastProneSupine/referenceState/boxModel/T4ANP/NH_D001/box_T4ANP_force.xml'
#    niftySimDisplacement = 'W:/philipsBreastProneSupine/referenceState/boxModel/T4ANP/NH_D001/U_box_T4ANP_force.txt'
#    febioASCIIFile       = 'W:/philipsBreastProneSupine/referenceState/boxModel/T4/FEBio/force_15_out.txt'
#    
#    comp_T4ANP_force = febioNiftySimComparison( niftysimModel, niftySimDisplacement, febioASCIIFile )
#    comp_T4ANP_force.plotNodalDisplacementHistogram( plotDir )
#    plt.close('all')
#    
#    # T4ANP grav
#    niftysimModel        = 'W:/philipsBreastProneSupine/referenceState/boxModel/T4ANP/NH_D001/box_T4ANP_grav.xml'
#    niftySimDisplacement = 'W:/philipsBreastProneSupine/referenceState/boxModel/T4ANP/NH_D001/U_box_T4ANP_grav.txt'
#    febioASCIIFile       = 'W:/philipsBreastProneSupine/referenceState/boxModel/T4/FEBio/grav_15_out.txt'
#    
#    comp_T4ANP_grav = febioNiftySimComparison( niftysimModel, niftySimDisplacement, febioASCIIFile )
#    comp_T4ANP_grav.plotNodalDisplacementHistogram( plotDir )
#    plt.close('all')
    
    
    
    #
    # H8 hgKappa experiments
    #
    
    # hgKappa 0.001
    niftysimModel        = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001_hgKVariation/box_H8_force_hgKappa0001.xml'
    niftySimDisplacement = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001_hgKVariation/U_box_H8_force_hgKappa0001.txt'
    febioASCIIFile       = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/FEBio/force_15_out.txt'
    
    comp_H8HG0001_force = febioNiftySimComparison( niftysimModel, niftySimDisplacement, febioASCIIFile )
    comp_H8HG0001_force.plotNodalDisplacementHistogram( plotDir, 7.0 )
    plt.close('all')
    
    # hgKappa 0.01
    niftysimModel        = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001_hgKVariation/box_H8_force_hgKappa001.xml'
    niftySimDisplacement = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001_hgKVariation/U_box_H8_force_hgKappa001.txt'
    febioASCIIFile       = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/FEBio/force_15_out.txt'
    
    comp_H8HG001_force = febioNiftySimComparison( niftysimModel, niftySimDisplacement, febioASCIIFile )
    comp_H8HG001_force.plotNodalDisplacementHistogram( plotDir, 7.0 )
    plt.close('all')
    
    # hgKappa 0.02
    niftysimModel        = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001_hgKVariation/box_H8_force_hgKappa002.xml'
    niftySimDisplacement = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001_hgKVariation/U_box_H8_force_hgKappa002.txt'
    febioASCIIFile       = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/FEBio/force_15_out.txt'
    
    comp_H8HG002_force = febioNiftySimComparison( niftysimModel, niftySimDisplacement, febioASCIIFile )
    comp_H8HG002_force.plotNodalDisplacementHistogram( plotDir, 7.0 )
    plt.close('all')
    
    # hgKappa 0.03
    niftysimModel        = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001_hgKVariation/box_H8_force_hgKappa003.xml'
    niftySimDisplacement = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001_hgKVariation/U_box_H8_force_hgKappa003.txt'
    febioASCIIFile       = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/FEBio/force_15_out.txt'
    
    comp_H8HG003_force = febioNiftySimComparison( niftysimModel, niftySimDisplacement, febioASCIIFile )
    comp_H8HG003_force.plotNodalDisplacementHistogram( plotDir, 7.0 )
    plt.close('all')
    
    
    # hgKappa 0.04
    niftysimModel        = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001_hgKVariation/box_H8_force_hgKappa004.xml'
    niftySimDisplacement = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001_hgKVariation/U_box_H8_force_hgKappa004.txt'
    febioASCIIFile       = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/FEBio/force_15_out.txt'
    
    comp_H8HG004_force = febioNiftySimComparison( niftysimModel, niftySimDisplacement, febioASCIIFile )
    comp_H8HG004_force.plotNodalDisplacementHistogram( plotDir, 7.0 )
    plt.close('all')
    
    
    
    # hgKappa 0.05
    niftysimModel        = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001_hgKVariation/box_H8_force_hgKappa005.xml'
    niftySimDisplacement = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001_hgKVariation/U_box_H8_force_hgKappa005.txt'
    febioASCIIFile       = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/FEBio/force_15_out.txt'
    
    comp_H8HG005_force = febioNiftySimComparison( niftysimModel, niftySimDisplacement, febioASCIIFile )
    comp_H8HG005_force.plotNodalDisplacementHistogram( plotDir, 7.0 )
    plt.close('all')
    
    
    
    # hgKappa 0.06
    niftysimModel        = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001_hgKVariation/box_H8_force_hgKappa006.xml'
    niftySimDisplacement = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001_hgKVariation/U_box_H8_force_hgKappa006.txt'
    febioASCIIFile       = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/FEBio/force_15_out.txt'
    
    comp_H8HG006_force = febioNiftySimComparison( niftysimModel, niftySimDisplacement, febioASCIIFile )
    comp_H8HG006_force.plotNodalDisplacementHistogram( plotDir, 7.0 )
    plt.close('all')
    
    
    # hgKappa 0.07
    niftysimModel        = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001_hgKVariation/box_H8_force_hgKappa007.xml'
    niftySimDisplacement = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001_hgKVariation/U_box_H8_force_hgKappa007.txt'
    febioASCIIFile       = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/FEBio/force_15_out.txt'
    
    comp_H8HG007_force = febioNiftySimComparison( niftysimModel, niftySimDisplacement, febioASCIIFile )
    comp_H8HG007_force.plotNodalDisplacementHistogram( plotDir, 7.0 )
    plt.close('all')
    
    
    # hgKappa 0.08
    niftysimModel        = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001_hgKVariation/box_H8_force_hgKappa008.xml'
    niftySimDisplacement = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001_hgKVariation/U_box_H8_force_hgKappa008.txt'
    febioASCIIFile       = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/FEBio/force_15_out.txt'
    
    comp_H8HG008_force = febioNiftySimComparison( niftysimModel, niftySimDisplacement, febioASCIIFile )
    comp_H8HG008_force.plotNodalDisplacementHistogram( plotDir, 7.0 )
    plt.close('all')
    
    
    # hgKappa 0.09
    niftysimModel        = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001_hgKVariation/box_H8_force_hgKappa009.xml'
    niftySimDisplacement = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001_hgKVariation/U_box_H8_force_hgKappa009.txt'
    febioASCIIFile       = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/FEBio/force_15_out.txt'
    
    comp_H8HG009_force = febioNiftySimComparison( niftysimModel, niftySimDisplacement, febioASCIIFile )
    comp_H8HG009_force.plotNodalDisplacementHistogram( plotDir, 7.0 )
    plt.close('all')
    
    
    # hgKappa 0.10
    niftysimModel        = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001_hgKVariation/box_H8_force_hgKappa010.xml'
    niftySimDisplacement = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001_hgKVariation/U_box_H8_force_hgKappa010.txt'
    febioASCIIFile       = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/FEBio/force_15_out.txt'
    
    comp_H8HG010_force = febioNiftySimComparison( niftysimModel, niftySimDisplacement, febioASCIIFile )
    comp_H8HG010_force.plotNodalDisplacementHistogram( plotDir, 7.0 )
    plt.close('all')
    
    
    # hgKappa 0.11
    niftysimModel        = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001_hgKVariation/box_H8_force_hgKappa011.xml'
    niftySimDisplacement = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001_hgKVariation/U_box_H8_force_hgKappa011.txt'
    febioASCIIFile       = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/FEBio/force_15_out.txt'
    
    comp_H8HG011_force = febioNiftySimComparison( niftysimModel, niftySimDisplacement, febioASCIIFile )
    comp_H8HG011_force.plotNodalDisplacementHistogram( plotDir, 7.0 )
    plt.close('all')
    
    
    # hgKappa 0.12
    niftysimModel        = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001_hgKVariation/box_H8_force_hgKappa012.xml'
    niftySimDisplacement = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/NH_D001_hgKVariation/U_box_H8_force_hgKappa012.txt'
    febioASCIIFile       = 'W:/philipsBreastProneSupine/referenceState/boxModel/H8/FEBio/force_15_out.txt'
    
    comp_H8HG012_force = febioNiftySimComparison( niftysimModel, niftySimDisplacement, febioASCIIFile )
    comp_H8HG012_force.plotNodalDisplacementHistogram( plotDir, 7.0 )
    plt.close('all')
    
    
    
    #
    # 
    #
    
    
    
    