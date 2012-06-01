#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import xmlModelReader as xReader
import os
import numpy as np
import modelDeformationVisualiser as defVis
import meshQualityTracker as mqt
from plotDoubleAxis import plotDoubleYAxis
from loadingFunction import loadingFunction




class convergenceAnalyser:
    
    
    ''' 
        This class looks at a single simulation and evaluates the information which is available 
    '''
    
    
    def __init__( self, xmlModelFileName, uFileName=None, eKinTotalFileName=None, eStrainTotalFileName=None ):
        ''' @param xmlModelFileName:     Path and name of the xml model (complete path)
            @param uFileName:            Name of the file holding the node displacements (niftySim default is U.txt)
            @param eKinTotalFileName:    Name of the file holding the total kinetic energy (niftySim default is EKinTotal.txt)
            @param eStrainTotalFileName: Name of the file holding the total strain energy (niftySim default is EStrainTotal.txt)
        '''
        
        #
        # Read the given xmlModel
        #
        self.xmlModelFileName = xmlModelFileName
        self.model            = xReader.xmlModelReader( self.xmlModelFileName )
        
        
        #
        # Extract those parameters which are relevant for the evaluation
        #
        
        # Output
        self._outputFreq = int( self.model.modelObject.Output.Freq ) 
        self._outputU            = False
        self._outputEKinTotal    = False
        self._outputEStrainTotal = False
        
        # System
        self.totalTime    = float( self.model.modelObject.SystemParams.TotalTime    )
        self.timeStep     = float( self.model.modelObject.SystemParams.TimeStep     )
        self.dampingCoeff = float( self.model.modelObject.SystemParams.DampingCoeff )
        
        
        if isinstance( self.model.modelObject.Output.Variable, list ):
            for var in self.model.modelObject.Output.Variable:
                if var == 'U':
                    self._outputU = True
                    continue
        
                if var == 'EKinTotal':
                    self._outputEKinTotal = True
                    continue
        
                if var == 'EStrainTotal':
                    self._outputEStrainTotal = True
        
        
        #
        # Require all 
        #
        if ( (self._outputU            == False) or 
             (self._outputEKinTotal    == False) or 
             (self._outputEStrainTotal == False) ):
            print( 'Not all information necessary for the evaluation are described in the model file. Add: ' )
            print('  <Output Freq="100">'  )
            print('    <Variable>'         )
            print('       U')
            print('    </Variable>')
            print('    <Variable>'         )
            print('       EKinTotal')
            print('    </Variable>')
            print('    <Variable>'         )
            print('       EStrainTotal')
            print('    </Variable>')
            print('  </Output>')
            print('To the model file.')

            return
        
        #
        # Try to find the output files...
        # First define the default names 
        #
        self.baseDir     = os.path.split( self.xmlModelFileName )[0]
        self.xmlBaseName = os.path.splitext( os.path.split( self.xmlModelFileName )[1] )[0]
        self.plotDir     = self.baseDir + '/plot/'
        
        
        if eKinTotalFileName == None:
            self.eKinTotalFileName = self.baseDir + '/EKinTotal_' + self.xmlBaseName + '.txt'
            print( 'Assuming kinetic energy to be at:' )
            print( '  ' + self.eKinTotalFileName )
        
        if eStrainTotalFileName == None:
            self.eStrainTotalFileName = self.baseDir + '/EStrainTotal_' + self.xmlBaseName + '.txt'
            print( 'Assuming strain energy to be at:' )
            print( '  ' + self.eStrainTotalFileName )
        
        if uFileName == None:
            self.uFileName = self.baseDir + '/U_' + self.xmlBaseName + '.txt'
            print( 'Assuming nodal displacements to be at:' )
            print( '  ' + self.uFileName )
        
        #
        # Check file existence
        #
        if ( ( not os.path.exists( self.eKinTotalFileName    ) ) or 
             ( not os.path.exists( self.eStrainTotalFileName ) ) or 
             ( not os.path.exists( self.uFileName            ) ) ):
            print( 'At least one file does not exist!' )
            return
        
        self.eKinTotal    = self._readEnergyFile( self.eKinTotalFileName    )
        self.eStrainTotal = self._readEnergyFile( self.eStrainTotalFileName )
        
        #
        # get the deformations and build the statistics...
        #
        self.vis                = defVis.modelDeformationVisualiser(self.model, self.uFileName )
        self.meshQualities = mqt.meshQualityTracker( modelDeformVis=self.vis )
        
        
        #
        # prepare plotting 
        # TODO: loadShape
        # TODO: check if kinetic and strain energy do have the same array-length
        #
        self.loadShape = 'POLY345FLAT4'
        self.evaluationIterations = self.eKinTotal.shape[0]
        self.loadingCurve, self.timeAxisVals  = loadingFunction(self.totalTime, self.loadShape, self.evaluationIterations)
        
        #
        # plot energies
        #
        self._plotMeshQualities()
        self._plotEnergies()
        
        
        
    def _plotEnergies( self ):
        
        if not os.path.exists( self.plotDir ):
            os.mkdir( self.plotDir )
        
        #
        # kinetic energy first
        #
        timeLabel     = '$t$'
        timeLabelUnit = '$t \;\mathrm{[s]}$'
        eKinLabel     = '$E_\mathrm{kin}$'
        eKinLabelUnit = '$E_\mathrm{kin}$'
        loadLabel     = '$L_\mathrm{rel}$'
        loadLabelUnit = '$L_\mathrm{rel}$'
        
        plotDoubleYAxis( xVals              = self.timeAxisVals, 
                         y1Vals             = self.eKinTotal, 
                         y2Vals             = self.loadingCurve, 
                         xLabel             = timeLabel, 
                         xLabelUnit         = timeLabelUnit, 
                         y1Label            = eKinLabel, 
                         y1LabelUnit        = eKinLabelUnit, 
                         y2Label            = loadLabel, 
                         y2LabelUnit        = loadLabelUnit, 
                         plotDirAndBaseName = self.plotDir + 'EKinTotal', 
                         printLegend        = False, 
                         y1Max              = 1.1 )
        
        eStrainLabel     = '$E_\mathrm{strain}$'
        eStrainLabelUnit = '$E_\mathrm{strain}$'
        
        plotDoubleYAxis( xVals              = self.timeAxisVals, 
                         y1Vals             = self.eStrainTotal, 
                         y2Vals             = self.loadingCurve, 
                         xLabel             = timeLabel, 
                         xLabelUnit         = timeLabelUnit, 
                         y1Label            = eStrainLabel, 
                         y1LabelUnit        = eStrainLabelUnit, 
                         y2Label            = loadLabel, 
                         y2LabelUnit        = loadLabelUnit, 
                         plotDirAndBaseName = self.plotDir + 'EKinTotal', 
                         printLegend        = False, 
                         y1Max              = 1.1 )



       
    def _plotMeshQualities( self ):
        
        self.meshQualities.getQualityMeasureResults('RadiusRatio')
        self.meshQualities.plotQualityMeasure( self.plotDir, self.loadShape, self.totalTime )
        
        self.meshQualities.getQualityMeasureResults('EdgeRatio')
        self.meshQualities.plotQualityMeasure( self.plotDir, self.loadShape, self.totalTime )
        
        self.meshQualities.getQualityMeasureResults( 'AspectFrobenius' )
        self.meshQualities.plotQualityMeasure( self.plotDir, self.loadShape, self.totalTime )
        
        pass
        


        
    def _readEnergyFile( self, energyFileName ):
        ''' Read the specified energy file 
            @param energyFileName:  Complete path to the file which holds the total energies (one entry per iteration)
            @return:                1-D numpy array with the deformations 
        '''
        
        fEnergy    = open( energyFileName    )
        dataEnergy = fEnergy.read()
        
        fEnergy.close()
        
        dataEnergy = dataEnergy.split()
        
        energy = []
        
        for dE in dataEnergy:
            try:
                energy.append( float( dE ) )
            except:
                print('Could not convert %s into float' % dE )
                continue
        
        return  np.array( energy ) 
       
        
        
        
if __name__ == '__main__':
    
    modelFileName = 'W:/philipsBreastProneSupine/referenceState/00_float_double/debug/modelFat_prone1G_it050000_totalTime05_poly345flat4.xml'
    analyzser = convergenceAnalyser( modelFileName )
    
    
    