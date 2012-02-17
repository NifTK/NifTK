
import os
import vtk
import nibabel as nib
import numpy as np
from vtk.util import numpy_support as VN


class maskFromSurface:
    
    
    
    
    def __init__( self, vtkPolyDataIn, niiFileNameIn, insideVal=255, outsideVal=0 ):
        ''' @param vtkPolyDataIn: Surface mesh which defines the inside
            @param niiFileNameIn: path of image file which will be used as a geometric reference 
            @param insideVal: pixel value inside the mesh
            @param outsideVal: pixel value outside the mesh 
        '''
        #
        # Check the input
        #
        if not os.path.exists( niiFileNameIn ) :
            print('Error: file ' + niiFileNameIn + ' does not exist...')
            return
        
        if not isinstance(vtkPolyDataIn, vtk.vtkPolyData ) :
            print( 'Error: Expecting vtkPolyData as input' )
            return
        
        #
        # Save to local vars
        #
        self.img           = nib.load( niiFileNameIn )#
        self.imgHdr        = self.img.get_header()
        self.imgDataShape  = self.imgHdr.get_data_shape()  
        self.surface       = vtkPolyDataIn
        
        self.outSideVal    = outsideVal
        self.insideVal     = insideVal
        
        self._generatePointList()
        self._checkPoints()
        
        
        
        
    def _generatePointList( self ):
        
        s = self.imgDataShape

        X,Y,Z = np.mgrid[ 0:s[0], 0:s[1], 0:s[2] ]
        
        self.pointList = np.vstack( ( X.reshape(-1), Y.reshape(-1), Z.reshape(-1), np.ones( s[0]*s[1]*s[2] ) ) )
        
        self.affine      = self.img.get_affine().copy()
        self.affine[0,0] = -self.affine[0,0]
        self.affine[1,1] = -self.affine[1,1]
        
        self.pointList = np.dot( self.affine, self.pointList ) 




    def _checkPoints( self ):
        
        l = self.imgDataShape[0] * self.imgDataShape[1] * self.imgDataShape[2]
        
        #
        # TODO: limit the test to the axis aligned bounding box!
        #
        pts = VN.vtk_to_numpy( self.surface.GetPoints().GetData() )
        
        m = np.min( pts, axis=0 )
        M = np.max( pts, axis=0 )
        
        idxmx = np.nonzero( self.pointList[0,:] > m[0]  )[0]
        idxmy = np.nonzero( self.pointList[1,:] > m[1]  )[0]
        idxmz = np.nonzero( self.pointList[2,:] > m[2]  )[0]
        
        idxMx = np.nonzero( self.pointList[0,:] < M[0]  )[0]
        idxMy = np.nonzero( self.pointList[1,:] < M[1]  )[0]
        idxMz = np.nonzero( self.pointList[2,:] < M[2]  )[0]
        
        idxList = np.lib.arraysetops.intersect1d( idxmx,   idxmy, True )
        idxList = np.lib.arraysetops.intersect1d( idxList, idxmz, True )
        idxList = np.lib.arraysetops.intersect1d( idxList, idxMx, True )
        idxList = np.lib.arraysetops.intersect1d( idxList, idxMy, True )
        idxList = np.lib.arraysetops.intersect1d( idxList, idxMz, True )
        
        selector = vtk.vtkSelectEnclosedPoints()
        selector.CheckSurfaceOn()
        selector.Initialize( self.surface )
        selector.SetTolerance(1e-5)
        
        self.mask = np.zeros( l, dtype=np.uint8 )
        self.mask[:] = self.outSideVal
        
        for i in idxList :
            
            if selector.IsInsideSurface(self.pointList[0,i], self.pointList[1,i], self.pointList[2,i] ) :
                self.mask[i] = self.insideVal
                
        self.mask = self.mask.reshape( (self.imgDataShape[0], self.imgDataShape[1], self.imgDataShape[2]) )




    def saveMaskToNii( self, strNiiMaskFileName ):
        
        maskImg = nib.Nifti1Image( self.mask, self.img.get_affine() )
        nib.save( maskImg, strNiiMaskFileName )




if __name__ == '__main__' :
    
    strFileName = 'W:/philipsBreastProneSupine/rigidAlignment/supine1kTransformCrop2Pad_zeroOrig.nii'
    strMesh     = 'W:/philipsBreastProneSupine/Meshes/meshMaterials5/pectWallSurf_impro_def.stl'

    stlReader = vtk.vtkSTLReader()
    stlReader.SetFileName(strMesh)
    stlReader.Update()
    
    pdMesh = stlReader.GetOutput()
    
    
    maskGen = maskFromSurface( pdMesh, strFileName, 0, 1 )
    maskGen.saveMaskToNii( strMesh.split('.')[0]  + '.nii' )
    
    
    
    