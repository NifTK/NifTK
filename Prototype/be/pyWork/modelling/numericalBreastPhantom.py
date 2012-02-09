#! /usr/bin/env python 
# -*- coding: utf-8 -*-

import numpy as np
import nibabel as nib
import commandExecution as cmdEx
import vtk2stl
import os
import stlBinary2stlASCII
import vtkVolMeshHandler as vmh
from getNodesCloseToMask import getNodesWithtinMask
import xmlModelGenerator as xmlGen
import materialSetGenerator as matGen


class numericalBreastPhantom:

    def __init__( self, outDir, edgeLength, meshlabSript=None, tetgenVol=75, tetgenQ=1.42, timeStep=1e-4, totalTime=1, damping=50 ):
        
        self.outDir               = outDir
        
        # images
        self.outNiiImageName      = outDir + 'vol.nii'
        self.outNiiChestImageName = outDir + 'volChestMask.nii'
        self.outNiiAirImageName   = outDir + 'volAirMask.nii'
        
        # meshes
        self.outSurfMeshSMesh     = outDir + 'surfMesh.smesh'
        self.outSurfMeshVTK       = outDir + 'surfMesh.vtk'
        self.outSurfMeshSTL       = outDir + 'surfMesh.stl'
        self.outVolMesh           = outDir + 'volMesh.smesh'
        self.outVolMesh           = self.outSurfMeshSTL.split('.')[0] + '.1.vtk'
        
        # models
        self.outXmlModelFat       = ''
        self.outXmlModelFatSkin   = ''
        self._outXmlModelFat      = outDir + 'modelFat.xml'         # base filenames so that the above can be reused
        self._outXmlModelFatSkin  = outDir + 'modelFatSkin.xml'     # 
        
        
        self.edgeLength           = edgeLength
        self.bottomPlateDiameter  = 170.0     # shape of the plate given in mm    
        self.meshlabScript        = meshlabSript
        self.tetgenVolume         = tetgenVol
        
        self.fatMaterialType      = 'NH'
        self.fatMaterialParams    = [ 100, 50000]
        self.skinMaterialType     = 'NH'
        self.skinMaterialParams   = [1000, 50000]
        
        self.totalTime            = totalTime
        self.timeStep             = timeStep
        self.damping              = damping
        
        
        self._breastVolIntensity  = 255.
        self._chestWallIntensity  = 64.
        
        
        self._createBellShapePhantomImageData()
        self._generateSurfaceMesh()
        
        if self.meshlabScript != None :
            self.outSurfMeshImproSTL = outDir + 'surfMeshImpro.stl'
            self.outVolMesh = self.outSurfMeshImproSTL.split('.')[0] + '.1.vtk'
            self._runMeshlabScript()

        self._generateVolumeMesh()
        self._prepareXMLFile()




    def _createBellShapePhantomImageData( self ):
        
        padSize = 5
        
        data     = np.zeros( (self.edgeLength, self.edgeLength, self.edgeLength) )
        originXY = (self.edgeLength - 1.) / 2.
        x, y     = np.meshgrid( range(self.edgeLength), range(self.edgeLength) )
        
        x = x - originXY
        y = y - originXY  
        
        z = np.e**( -(x**2 + y**2) / (2.*(self.edgeLength / 4.)**2. ) ) * (self.edgeLength - 2.)/2. + padSize
        m = np.zeros_like( z )
        m[ np.nonzero( np.sqrt( (x**2. + y**2.)  )  < ( (self.edgeLength - padSize) / 2.) ) ] = 1.0
        
        for iZ in range( padSize ):
            d = np.zeros_like( z )
            d = self._chestWallIntensity * m
            data[:,:,iZ] = d
            
        for iZ in range( padSize, self.edgeLength-1 ) :
            d = np.zeros_like( z )
            d[ np.nonzero( z >=iZ ) ] = self._breastVolIntensity
            d = d * m
            data[:,:,iZ] = d
        #
        # Assumption of the size of the numerical model
        #   the edgeLength (diameter of the retromammary surface) is about 15cm
        #
        scale       = self.bottomPlateDiameter / self.edgeLength
        affine = np.eye(4)
        affine[0,0] = -scale
        affine[1,1] = -scale
        affine[2,2] =  scale
        niiImageOut = nib.Nifti1Image( np.array(data, np.uint8 ), affine )
        nib.save( niiImageOut, self.outNiiImageName )
        
        #
        # Create mask image for "pectoral muscle" -> will be fixed
        #   -> niftkThreshold -i vol.nii -l 60 -u 70 -in 255 -out 0 -o pectVolMask.nii
        #
        threshCmd     = 'niftkThreshold'
        threshParams  = ' -i ' + self.outNiiImageName 
        threshParams += ' -o ' + self.outNiiChestImageName 
        threshParams += ' -l ' + str( '%i' % (self._chestWallIntensity-5) ) 
        threshParams += ' -u ' + str( '%i' % (self._chestWallIntensity+5) ) 
        threshParams += ' -in 255 ' 
        threshParams += ' -out 0 '  
        
        cmdEx.runCommand( threshCmd, threshParams )
        
        #
        # Dilate the mask image
        #
        dilateCmd     = 'niftkDilate'
        dilateParams  = ' -i ' + self.outNiiChestImageName 
        dilateParams += ' -o ' + self.outNiiChestImageName
        dilateParams += ' -r 2' 
        dilateParams += ' -it 4' 
        dilateParams += ' -d 255 ' 
        dilateParams += ' -b 0 '
        cmdEx.runCommand( dilateCmd, dilateParams )
         
          
        # Create mask image for "air" -> will be fixed
        threshCmd     = 'niftkThreshold'
        threshParams  = ' -i ' + self.outNiiImageName 
        threshParams += ' -o ' + self.outNiiAirImageName 
        threshParams += ' -l ' + str( '%i' % ( 0) ) 
        threshParams += ' -u ' + str( '%i' % ( 5) ) 
        threshParams += ' -in 255 ' 
        threshParams += ' -out 0 '  
        cmdEx.runCommand( threshCmd, threshParams )
        
        # Dilate the mask image
        dilateCmd     = 'niftkDilate'
        dilateParams  = ' -i ' + self.outNiiAirImageName
        dilateParams += ' -o ' + self.outNiiAirImageName
        dilateParams += ' -r 2' 
        dilateParams += ' -it 4' 
        dilateParams += ' -d 255 ' 
        dilateParams += ' -b 0 '
        cmdEx.runCommand( dilateCmd, dilateParams )
          
        
        
        
    def _generateSurfaceMesh( self ):
        
        surfGenCommand = 'medsurfer'
        
        #
        # Parameters used for meshMaterial
        #
        surfGenParams  = ' -iso 128 '      
        surfGenParams += ' -df 0.8 '       
        surfGenParams += ' -shrink 2 2 2 '
        surfGenParams += ' -presmooth'
        surfGenParams += ' -niter 40'
        surfGenParams += ' -img '  + self.outNiiImageName 
        surfGenParams += ' -surf ' + self.outSurfMeshSMesh
        surfGenParams += ' -vtk '  + self.outSurfMeshVTK
        
        cmdEx.runCommand( surfGenCommand, surfGenParams )
        vtk2stl.vtk2stl([self.outSurfMeshVTK])
        



    def _runMeshlabScript( self ):
        
        meshLabCommand         = 'meshlabserver'
        meshLabParamrs         = ' -i ' + self.outSurfMeshSTL
        meshLabParamrs        += ' -o ' + self.outSurfMeshImproSTL
        meshLabParamrs        += ' -s ' + self.meshlabScript
        
        cmdEx.runCommand( meshLabCommand, meshLabParamrs )
        stlBinary2stlASCII.stlBinary2stlASCII( self.outSurfMeshImproSTL )




    def _generateVolumeMesh( self ):
        
        curDir = os.getcwd()
        os.chdir( self.outDir )
        volMeshCommand = 'tetgen'
        
        if self.meshlabScript == None : 
            volMeshParams  = ' -pq1.42a' + str( self.tetgenVolume ) + 'K ' + self.outSurfMeshSTL 
        else :
            volMeshParams  = ' -pq1.42a' + str( self.tetgenVolume ) + 'K ' + self.outSurfMeshImproSTL
            
        cmdEx.runCommand(volMeshCommand, volMeshParams )
        os.chdir( curDir )



        
    def _prepareXMLFile( self ):
        ''' Execute those parts, which need to be executed once only.
        '''
        
        #
        # find the fixed nodes
        #
        self.mesh = vmh.vtkVolMeshHandler( self.outVolMesh )
        
        ( self.ptsFixChest, self.idxFixChest ) = getNodesWithtinMask( self.outNiiChestImageName, 128., 
                                                                      self.mesh.volMeshPoints, 
                                                                      self.mesh.surfMeshPoints )
        
        #
        # find the skin element nodes
        #
        self.materialGen = matGen.materialSetGenerator( self.mesh.volMeshPoints, 
                                                        self.mesh.volMeshCells, 
                                                        self.outNiiImageName, 
                                                        self.outNiiAirImageName, 
                                                        self.outVolMesh, 255, 13, 14, 3, 
                                                        chestWallMaskImage=None )
        
    
    
    
    def generateXMLmodelFatOnly( self, gravityVector = [0., 0., -1. ], gravityMagnitude = 20, 
                                 fileIdentifier=None, extMeshNodes=None ):
        ''' @summary: Generates the xml model
            @param gravityVector: Direction of gravity
            @param gravityMagnitude: Assumed gravitational acceleration
            @param extMeshNodes: np-Array with the mesh nodes. Must be valid for the mesh generated 
                                 within this class. Assumed to be given in mm (millimetre).
            @param fileIdentifier: Extension which is used to specify the model file. 
            @return: the xmlModelGenerator Instance
        '''
        
        if not os.path.exists(self.outVolMesh) :
            print('Error: Surface mesh does not exists')
            return
        
        
        if fileIdentifier != None :
            self.outXmlModelFat = self._outXmlModelFat.split('.xm')[0] + str( fileIdentifier ) + '.xml'
        else :
            self.outXmlModelFat = self._outXmlModelFat
        
        self.mesh = vmh.vtkVolMeshHandler( self.outVolMesh )
               
        if (extMeshNodes == None) :
            fatGen = xmlGen.xmlModelGenrator( self.mesh.volMeshPoints/1000., self.mesh.volMeshCells[ : , 1:5], 'T4ANP')
        else :
            fatGen = xmlGen.xmlModelGenrator( extMeshNodes/1000., self.mesh.volMeshCells[ : , 1:5], 'T4ANP')
        
        fatGen.setFixConstraint( self.idxFixChest, 0 )
        fatGen.setFixConstraint( self.idxFixChest, 1 )
        fatGen.setFixConstraint( self.idxFixChest, 2 )
        
        fatGen.setMaterialElementSet( 'NH', 'FAT', [100, 50000], fatGen.allElemenstArray )
        
        fatGen.setGravityConstraint( gravityVector, gravityMagnitude, fatGen.allNodesArray, 'RAMP' )
        fatGen.setOutput( 5000, 'U' )
        fatGen.setSystemParameters( timeStep           = self.timeStep, 
                                    totalTime          = self.totalTime, 
                                    dampingCoefficient = self.damping, 
                                    hgKappa = 0.05, density = 1000 )    
        fatGen.writeXML( self.outXmlModelFat )
        
        return fatGen




    def generateXMLmodel( self, gravityVector = [0., 0., -1. ], gravityMagnitude = 20, 
                          fileIdentifier=None, extMeshNodes=None, skin=True ):
        ''' @summary: Generates the xml model
            @param gravityVector: Direction of gravity
            @param gravityMagnitude: Assumed gravitational acceleration
            @param extMeshNodes: np-Array with the mesh nodes. Must be valid for the mesh generated 
                                 within this class. Assumed to be given in mm (millimetre).
            @param fileIdentifier: Extension which is used to specify the model file. 
            @return: the xmlModelGenerator Instance
        '''
        
        if not os.path.exists(self.outVolMesh) :
            print('Error: Surface mesh does not exists')
            return
        
        
        if fileIdentifier != None :
            self.outXmlModelFat = self._outXmlModelFat.split('.xm')[0] + str( fileIdentifier ) + '.xml'
        else :
            self.outXmlModelFat = self._outXmlModelFat
        
        self.mesh = vmh.vtkVolMeshHandler( self.outVolMesh )
               
        if (extMeshNodes == None) :
            gen = xmlGen.xmlModelGenrator( self.mesh.volMeshPoints/1000., self.mesh.volMeshCells[ : , 1:5], 'T4ANP')
        else :
            gen = xmlGen.xmlModelGenrator( extMeshNodes/1000., self.mesh.volMeshCells[ : , 1:5], 'T4ANP')
        
        gen.setFixConstraint( self.idxFixChest, 0 )
        gen.setFixConstraint( self.idxFixChest, 1 )
        gen.setFixConstraint( self.idxFixChest, 2 )
        
        
        if skin:
            # Case skin and fat 
            gen.setMaterialElementSet( self.skinMaterialType, 'SKIN', self.skinMaterialParams, self.materialGen.skinElements )
            gen.setMaterialElementSet( self.fatMaterialType,  'FAT',  self.fatMaterialParams,  self.materialGen.fatElemetns  )

        else :
            # Case: fat only 
            gen.setMaterialElementSet( self.fatMaterialType, 'FAT', self.fatMaterialParams, gen.allElemenstArray )
        
        gen.setGravityConstraint( gravityVector, gravityMagnitude, gen.allNodesArray, 'RAMP' )
        gen.setOutput( 5000, 'U' )
        gen.setSystemParameters( timeStep           = self.timeStep, 
                                    totalTime          = self.totalTime, 
                                    dampingCoefficient = self.damping, 
                                    hgKappa = 0.05, density = 1000 )    
        gen.writeXML( self.outXmlModelFat )
        
        return gen

        


if __name__ == '__main__':
    
    outPath    = 'C:/data/test/'
    edgeLength = 400
    mlxFile    = 'W:/philipsBreastProneSupine/Meshes/mlxFiles/surfProcessing_coarse.mlx'
    phantom    = numericalBreastPhantom( outPath, edgeLength, mlxFile )
    phantom.generateXMLmodelFatOnly()
    
