#! /usr/bin/env python 
# -*- coding: utf-8 -*-


from rigidHelpers import rotMatZ
import numpy as np
from matplotlib.pyplot import imshow
from mayaviPlottingWrap import plotArrayAs3DPoints
from conversions import numpyArrayToStr
from mayavi import mlab
import nibabel as nib
import os
import xmlModelGenerator as xGen
import commandExecution as cmdEx
import vtk2stl
import stlBinary2stlASCII


class testImageGenerator :
    ''' class to generate a test image with a square rotated by 
    '''
    
    
    
    
    def __init__(self, imageSize, squareSize, rotationAngle ):
        '''
            @param imageSize:     Numpy array (size 3x1) with image size.
            @param squareSize:    Numpy array (size 3x1) with size of the square which
                                  will be centered in the image
            @param rotationAngle: Angle around which the square will be rotated (in radian)
        '''
        
        # image contents related vars
        self.imageSize     = imageSize
        self.squareSize    = squareSize
        self.rotationAngle = rotationAngle
        
        # image geometry related vars
        self.imgOrigin  = np.array((0., 0., 0.))
        self.imgSpacing = np.array((1., 1., 1.))
        
        self._generateData()
        
        
        
        
    def _generateData( self ):
        
        centre    = self.imageSize / 2.
        x, y      = np.meshgrid( range( self.imageSize[0] ), 
                                 range( self.imageSize[1] ) )
        
        x = x - centre[0]
        y = y - centre[1]
        
        # Now rotate the coordinates around the centre of course...
        Rz = rotMatZ( -self.rotationAngle )
        
        xP = x.reshape( (1,-1) )
        yP = y.reshape( (1,-1) )
        zP = np.ones_like( xP )
        
        pos  = np.vstack( (xP, yP, zP, zP) )
        posP = np.dot( Rz, pos )
        
        
        xPP = posP[0,:].reshape( self.imageSize[0:2] )
        yPP = posP[1,:].reshape( self.imageSize[0:2] )
        
        # from here on it is a simple thresholding
        
        # generate the in-plane data
        self.xyData = np.zeros( (self.imageSize[0], self.imageSize[1]), np.int8 )
        self.xyData[ (xPP < self.squareSize[0] / 2.) & (xPP > -self.squareSize[0] / 2.) & 
                     (yPP < self.squareSize[1] / 2.) & (yPP > -self.squareSize[1] / 2.)   ] = 255
        
        self.data    = np.zeros( self.imageSize, np.uint8 )
        self.dataLMS = np.zeros( self.imageSize, np.uint8 )
        
        startIdx = int((self.imageSize[2] - self.squareSize[2]) / 2.)   
        endIdx = startIdx + self.squareSize[2]
        
        # fill in-plane data into volume data
        for i in range(startIdx, endIdx ):
            self.data[:,:,i] = self.xyData        
        
        
        # Which points should be kept as landmarks?
        #   all x, lowest y
        dilateSize = 2
        bLMS       = (xPP < self.squareSize[0] / 2.) & (xPP >  -self.squareSize[0] / 2.) & (yPP < -self.squareSize[1] / 2. + 1.0 ) & (yPP > (-self.squareSize[1] / 2.) ) 
        self.xyLMS = np.zeros( (self.imageSize[0], self.imageSize[1]), np.int8 )
        self.xyLMS[ bLMS ] = 255        
        
        
        # dilated image of the landmarks
        bLMSdilate= ( ( xPP < (  self.squareSize[0] / 2. + dilateSize ) ) & 
                      ( xPP > ( -self.squareSize[0] / 2. - dilateSize ) ) &
                      ( yPP < ( -self.squareSize[1] / 2. + 1.0 + dilateSize ) ) & 
                      ( yPP > ( -self.squareSize[1] / 2. - dilateSize) ) )

        self.xyLMSDilate = np.zeros( (self.imageSize[0], self.imageSize[1]), np.int8 )
        self.xyLMSDilate[ bLMSdilate ] = 255        
        
        # fill in-plane data into volume data
        for i in range(startIdx-dilateSize, endIdx + dilateSize ):
            self.dataLMS[:,:,i] = self.xyLMSDilate
        
        
        # real world coordinates
        xRW, yRW, zRW = np.mgrid[ 0:self.imageSize[0], 0:self.imageSize[1], 0:self.imageSize[2] ]
        w = np.ones_like( xRW )
        
        self.affine = np.eye( 4 )
        self.affine[0:3,0:3] = np.diag(self.imgSpacing )
        self.affine[0:3,3]   = self.imgOrigin
        
        pRW = np.dot(self.affine, np.vstack( ( xRW.reshape((1,-1)), 
                                          yRW.reshape((1,-1)), 
                                          zRW.reshape((1,-1)), 
                                          w.reshape  ((1,-1)))))
        
        xRW = pRW[0,:].reshape( xRW.shape )
        yRW = pRW[1,:].reshape( yRW.shape )
        zRW = pRW[2,:].reshape( zRW.shape )
        
        xLMSCds = []
        yLMSCds = []
        zLMSCds = []
        
        for i in range(startIdx, endIdx ):
            xRWtemp = xRW[:,:,i] # this does not change over iterations
            yRWtemp = yRW[:,:,i] # this does not change over iterations
            zRWtemp = zRW[:,:,i] # this is a constant within the iterations...
            
            xLMSCds.append( xRWtemp[bLMS] )
            yLMSCds.append( yRWtemp[bLMS] )
            zLMSCds.append( zRWtemp[bLMS] )
            
        xLMSCds = np.array(xLMSCds).reshape(-1,1)
        yLMSCds = np.array(yLMSCds).reshape(-1,1)
        zLMSCds = np.array(zLMSCds).reshape(-1,1)
        self.landmarkCds = np.squeeze( np.array((xLMSCds, yLMSCds, zLMSCds)).T )
        
        print('done')
            
        
        
        
    def writeLandmarkFile(self, fileName) :
        
        self.landmarkFileName = fileName
        f = file(fileName, 'w')
        f.write(numpyArrayToStr(self.landmarkCds, True, '' ))
        f.close()




    def writeImageFile(self, imageFileName) :
        
        self.imageFileName         = imageFileName
        self.imageLandmakrFileName = imageFileName.split('.nii')[0] + '_lmsDilate.nii' 
        A = self.affine.copy()
        A = np.dot( rotMatZ(np.pi), A )
        niiImageOut = nib.Nifti1Image( np.array(self.data, np.uint8 ), A )
        nib.save( niiImageOut, imageFileName )

        # Save the landmarks
        niiLMSImageOut = nib.Nifti1Image( np.array(self.dataLMS, np.uint8 ), A )
        nib.save( niiLMSImageOut, self.imageLandmakrFileName )



    def buildNiftySimModel( self, xmlFileName, mlxFile = None ):
        
        #
        # WARNING: Consider Mesh generation from data as this is a really simple example!
        #
        
        self.xmlFileName = xmlFileName
        self.surfSmesh    = self.imageFileName.split('.nii')[0] + '.smesh'
        self.surfVTK      = self.imageFileName.split('.nii')[0] + '.vtk'
        #
        # Parameters used for meshMaterial
        #
        medSurferParms  = ' -iso 80 '      
        medSurferParms += ' -df 0.8 '       
        #medSurferParms += ' -shrink 2 2 2 '
        #medSurferParms += ' -presmooth'
        #medSurferParms += ' -niter 40'
        
        
        # Build the soft tissue mesh:
        
        medSurfBreastParams  = ' -img '  + self.imageFileName 
        medSurfBreastParams += ' -surf ' + self.surfSmesh
        medSurfBreastParams += ' -vtk '  + self.surfVTK
        medSurfBreastParams += medSurferParms
        
        cmdEx.runCommand( 'medSurfer', medSurfBreastParams )
        
        
        surfMeshSTL      = vtk2stl.vtk2stl( [self.surfVTK] )
        surfMeshImproSTL = surfMeshSTL[0].split('.stl')[0] + '_impro.stl'
        meshLabParams         = ' -i ' + surfMeshSTL[0]
        meshLabParams        += ' -o ' + surfMeshImproSTL
        if mlxFile != None :
            meshLabParams    += ' -s ' + mlxFile
            
        meshLabCommand        = 'meshlabserver' 
        cmdEx.runCommand( meshLabCommand, meshLabParams )

        # Surface mesh generation XXX
        # Volume mesh generation
        # Build xml file
        #   - fix constraints?
        curDir = os.getcwd()
        meshdir = os.path.dirname(self.imageFileName)
        
        # Build the volume mesh
        
        os.chdir(meshdir)
        
        tetCmd = 'tetgen'
        # convert the output file to ASCII format
        stlBinary2stlASCII.stlBinary2stlASCII( surfMeshImproSTL )
        
        
        # build the volume mesh
        tetVolParams = ' -pq1.42a50K ' + surfMeshImproSTL
        cmdEx.runCommand( 'tetgen', tetVolParams )        
        
        os.chdir(curDir)
        
        


        
    def showXYPlane(self):
        imshow( self.xyData, interpolation ='nearest' )
        
        
        
        
    def showXYLMS(self):
        imshow(self.xyLMS, interpolation ='nearest')
        
        
        
        
    def showLMSin3D(self):
        plotArrayAs3DPoints(self.landmarkCds)    
        p=mlab.pipeline.scalar_field(self.data)
        p.spacing = np.array( (self.affine[0,0],self.affine[1,1],self.affine[2,2]) )
        p.origin  = np.array( (self.affine[0,3],self.affine[1,3],self.affine[2,3]) ) 
        mlab.pipeline.volume( p )
        #mlab.contour3d(self.data)
    
    
    
        
if __name__ == '__main__':
    
    imgS = np.array( (120,120,120) )
    sqS  = np.array( (60,60,60) )
    phi  = 45. * np.pi/180.0
    
    #niiFileOut = 'W:/philipsBreastProneSupine/feirGravTest/input/square_120_60_r45.nii'
    #lmFileOut  = 'W:/philipsBreastProneSupine/feirGravTest/input/square_120_60_r45_LM.txt'
    
    niiFileOut = 'D:/data/test/square_120_60_r45.nii'
    lmFileOut  = 'D:/data/test/square_120_60_r45_LM.txt'
    xmlModel   = 'D:/data/test/square_120_60_r45.xml'
    
    mlxSurfImproFile = 'D:/data/copiedFromUCL/mlxFiles/uniformRemeshing.mlx'
    
    tig = testImageGenerator(imgS, sqS, phi)
    tig.showXYLMS()    
    tig.showLMSin3D()
    tig.writeImageFile(niiFileOut) 
    tig.writeLandmarkFile(lmFileOut)
    tig.buildNiftySimModel(xmlModel, mlxSurfImproFile)
    



