#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import xmlModelReader as xReader
import os
import numpy as np
import modelDeformationVisualiser as defVis
import meshQualityTracker as mqt
from plotDoubleAxis import plotDoubleYAxis
from loadingFunction import loadingFunction
from pyPdf import PdfFileWriter, PdfFileReader
from scipy.stats import scoreatpercentile


class convergenceAnalyser:
    
    
    ''' 
        This class looks at a single simulation and evaluates the information available 
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
        self._outputFreq         = int( self.model.modelObject.Output.Freq ) 
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
        # Constraints
        # Try to find the loading shape...
        #
        self.loadShape = 'POLY345FLAT4'
        
        if isinstance( self.model.modelObject.Constraint, list ):
            for var in self.model.modelObject.Constraint:
                if var.LoadShape != None:
                    self.loadShape = var.LoadShape
                
        print( 'Using load shape: ' + self.loadShape )
        
        #
        # Require all possible outputs
        #
        if ( (self._outputU            == False) or 
             (self._outputEKinTotal    == False) or 
             (self._outputEStrainTotal == False) ):
            print( 'Not all information necessary for the evaluation are described in the model file. Add: ' )
            print('  <Output Freq="100">'  )
            print('    <Variable>'         )
            print('       U'               )
            print('    </Variable>'        )
            print('    <Variable>'         )
            print('       EKinTotal'       )
            print('    </Variable>'        )
            print('    <Variable>'         )
            print('       EStrainTotal'    )
            print('    </Variable>'        )
            print('  </Output>'            )
            print('To the model file.'     )

            return
        
        #
        # Try to find the output files...
        # First define the default names 
        #
        self.baseDir     = os.path.split( self.xmlModelFileName )[0]
        self.xmlBaseName = os.path.splitext( os.path.split( self.xmlModelFileName )[1] )[0]
        self.plotDir     = self.baseDir + '/plot_' + self.xmlBaseName + '/'
        
        
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
        # get the deformations and build the statistics.
        # then get the quality measures which were evaluated 
        #
        self.vis             = defVis.modelDeformationVisualiser(self.model, self.uFileName )
        self.meshQualities   = mqt.meshQualityTracker( modelDeformVis = self.vis )
        self.qualityMeasures = self.meshQualities.qualityMeasures
        
        #
        # Calculate mean displacements
        #
        numSamples = len( self.vis.deformedNodes )
        self.meanDisplacements = np.zeros( numSamples )
        
        for i in range( numSamples ):
            self.meanDisplacements[i] = np.mean( np.sqrt( ( self.vis.displacements[i][:,0] * self.vis.displacements[i][:,0] ) + 
                                                          ( self.vis.displacements[i][:,1] * self.vis.displacements[i][:,1] ) +
                                                          ( self.vis.displacements[i][:,2] * self.vis.displacements[i][:,2] )   ) )   
        
        
        #
        # prepare plotting 
        #
        self.evaluationIterations = self.eKinTotal.shape[0]
        self.loadingCurve, self.timeAxisVals  = loadingFunction( self.totalTime, self.loadShape, self.evaluationIterations )
        
        #
        # plot energies
        #
        self._plotMeshQualities()
        self._plotEnergiesAndMeanDisp()
        self._combinePdfs()
        
        
        
        
    def _plotEnergiesAndMeanDisp( self ):
        
        if not os.path.exists( self.plotDir ):
            os.mkdir( self.plotDir )

        
        #
        # kinetic energy 
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
                         y2Max              = 1.1 )
        
        #
        # strain energy 
        #
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
                         plotDirAndBaseName = self.plotDir + 'EStrainTotal', 
                         printLegend        = False, 
                         y2Max              = 1.1 )
        
        
        #
        # fraction strain over kinetic energy 
        #
        
        per95 = scoreatpercentile( self.eKinTotal / self.eStrainTotal, 95 )
        
        eKinOverStrainLabel     = '$E_\mathrm{kin} / E_\mathrm{strain} $'
        eKinOverStrainLabelUnit = '$E_\mathrm{kin} / E_\mathrm{strain} $'
        
        plotDoubleYAxis( xVals              = self.timeAxisVals, 
                         y1Vals             = self.eKinTotal / self.eStrainTotal , 
                         y2Vals             = self.loadingCurve, 
                         xLabel             = timeLabel, 
                         xLabelUnit         = timeLabelUnit, 
                         y1Label            = eKinOverStrainLabel, 
                         y1LabelUnit        = eKinOverStrainLabelUnit, 
                         y2Label            = loadLabel, 
                         y2LabelUnit        = loadLabelUnit, 
                         plotDirAndBaseName = self.plotDir + 'EKinOverStrainTotal', 
                         printLegend        = False, 
                         y1Max              = 1.1 * per95,
                         y2Max              = 1.1 )
        
        
        #
        # mean displacement 
        #
        meanDispLabel     = '$\overline{ \| u \| }$'
        meanDispLabelUnit = '$\overline{ \| u \| } \; \mathrm{[mm]}$'
        
        plotDoubleYAxis( xVals              = self.timeAxisVals, 
                         y1Vals             = self.meanDisplacements,
                         y2Vals             = self.loadingCurve, 
                         xLabel             = timeLabel, 
                         xLabelUnit         = timeLabelUnit, 
                         y1Label            = meanDispLabel, 
                         y1LabelUnit        = meanDispLabelUnit, 
                         y2Label            = loadLabel, 
                         y2LabelUnit        = loadLabelUnit, 
                         plotDirAndBaseName = self.plotDir + 'MeanDisplacement', 
                         printLegend        = False, 
                         y2Max              = 1.1 )
        
        


    def _plotMeshQualities( self ):
        
        if not os.path.exists( self.plotDir ):
            os.mkdir( self.plotDir )
            
        for qm in self.qualityMeasures:
            self.meshQualities.plotQualityMeasure( self.plotDir, self.loadShape, self.totalTime, qm )
        
        


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
       
        
        
        
    def _combinePdfs( self ):
        
        pdfsIn = []
        files  = []
        
        # mean displacemen
        files.append(file( self.plotDir + 'MeanDisplacement.pdf', 'rb' ) )
        pdfsIn.append( PdfFileReader( files[-1] ) )
        
        # Energies
        files.append( file( self.plotDir + 'EKinTotal.pdf', 'rb' ) )
        pdfsIn.append( PdfFileReader( files[-1] ) )

        files.append( file( self.plotDir + 'EStrainTotal.pdf', 'rb' ) )
        pdfsIn.append( PdfFileReader( files[-1] ) )
        
        files.append( file( self.plotDir + 'EKinOverStrainTotal.pdf', 'rb' ) )
        pdfsIn.append( PdfFileReader( files[-1] ) )
        
        # 
        for qm in self.qualityMeasures:
            files.append( file( self.plotDir + qm + '.pdf', 'rb' ) )
            pdfsIn.append( PdfFileReader( files[-1] ) )
            
        output = PdfFileWriter()
        
        for pFile in pdfsIn:
            output.addPage( pFile.getPage( 0 ) )
        
        outputStream = file( self.plotDir + "CombinedInfo.pdf", "wb" )
        output.write( outputStream )
        outputStream.close()
    
        # close the input files
        for f in files:
            f.close()
    
    
    
    
if __name__ == '__main__':
    
    modelFileName = 'W:/philipsBreastProneSupine/referenceState/00_float_double/debug/modelFat_prone1G_it050000_totalTime05_poly345flat4.xml'
    analyzser = convergenceAnalyser( modelFileName )
    
    
    