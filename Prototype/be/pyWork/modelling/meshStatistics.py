#! /usr/bin/env python 
# -*- coding: utf-8 -*-


import numpy as np
import vtkMeshFileReader as vmr
import matplotlib.pyplot as plt
import vtk
from vtk.util import numpy_support as VN


class meshStatistics:
    ''' Class which builds mesh statistics 
     
    '''
    
    def __init__( self, nodes, elements ):
        
        self.nodes    = nodes
        self.elements = elements
        
        self.qualityMeasures = {}
        
        self._buildVTKMesh()
        self._calcQualityMeasures()
        self._calcBasicStatistics()
        pass
    
    
    
    
    def _buildVTKMesh( self ):
        
        
        self.unstructuredGrid = vtk.vtkUnstructuredGrid()
        pts = vtk.vtkPoints()
        pts.SetData( VN.numpy_to_vtk(self.nodes, deep=True) )
        self.unstructuredGrid.SetPoints( pts )
        
        #
        # generate cells
        #
        
        for i in range( self.elements.shape[0] ):
            tet = vtk.vtkTetra()
            tet.GetPointIds().SetId(0, self.elements[ i, 0 ])
            tet.GetPointIds().SetId(1, self.elements[ i, 1 ])
            tet.GetPointIds().SetId(2, self.elements[ i, 2 ])    
            tet.GetPointIds().SetId(3, self.elements[ i, 3 ])
            self.unstructuredGrid.InsertNextCell(tet.GetCellType(), tet.GetPointIds())
            
#        aTetraMapper = vtk.vtkDataSetMapper()
#        aTetraMapper.SetInput( self.unstructuredGrid )
#        aTetraActor = vtk.vtkActor()
#        aTetraActor.SetMapper(aTetraMapper)
#        aTetraActor.AddPosition(4, 0, 0)
#        aTetraActor.GetProperty().SetDiffuseColor(0, 1, 0)
#            
#        ren = vtk.vtkRenderer()
#        renWin = vtk.vtkRenderWindow()
#        renWin.AddRenderer(ren)
#        renWin.SetSize(300, 150)
#        iren = vtk.vtkRenderWindowInteractor()
#        iren.SetRenderWindow(renWin)
#        
#        ren.SetBackground(.1, .2, .4)
#        
#        ren.AddActor(aTetraActor)
#        ren.ResetCamera()
#        ren.GetActiveCamera().Azimuth(30)
#        ren.GetActiveCamera().Elevation(20)
#        ren.GetActiveCamera().Dolly(2.8)
#        ren.ResetCameraClippingRange()
#        
#        # Render the scene and start interaction.
#        iren.Initialize()
#        renWin.Render()
#        iren.Start()
        
        
        
        
        
    def _calcQualityMeasures( self ):
        
        
        #
        # iterate through all the different mesh quality measures and store these in the 
        # dictionary self.qualityMeasures
        #
        
        
#      X  SetTetQualityMeasureToAspectBeta 
#      X  SetTetQualityMeasureToAspectFrobenius 
#      X  SetTetQualityMeasureToAspectGamma 
#      X  SetTetQualityMeasureToAspectRatio 
#      X  SetTetQualityMeasureToCollapseRatio 
#      X  SetTetQualityMeasureToCondition 
#      X  SetTetQualityMeasureToDistortion 
#      X  SetTetQualityMeasureToEdgeRatio 
#      X  SetTetQualityMeasureToJacobian 
#      X  SetTetQualityMeasureToMinAngle 
#      X  SetTetQualityMeasureToRadiusRatio 
#      X  SetTetQualityMeasureToRelativeSizeSquared 
#      X  SetTetQualityMeasureToScaledJacobian 
#      X  SetTetQualityMeasureToShape 
#      X  SetTetQualityMeasureToShapeAndSize 
#      X  SetTetQualityMeasureToVolume 

        
        self.meshQualityFilter = vtk.vtkMeshQuality()
        self.meshQualityFilter.SetInput( self.unstructuredGrid )
        
        
        #
        # Radius Ratio
        #        
        self.meshQualityFilter.SetTetQualityMeasureToRadiusRatio()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['RadiusRatio'] = np.array( qualityMeasure, copy=True )

        
        #
        # Min Angle
        #        
        self.meshQualityFilter.SetTetQualityMeasureToMinAngle()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['MinAngle'] = np.array( qualityMeasure, copy=True )

        
        #
        # Edge Ratio
        #        
        self.meshQualityFilter.SetTetQualityMeasureToEdgeRatio()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['EdgeRatio'] = np.array( qualityMeasure, copy=True )

        
        #
        # Jacobian
        #        
        self.meshQualityFilter.SetTetQualityMeasureToJacobian()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['Jacobian'] = np.array( qualityMeasure, copy=True )

        
        #
        # Scaled Jacobian
        #        
        self.meshQualityFilter.SetTetQualityMeasureToScaledJacobian()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['ScaledJacobian'] = np.array( qualityMeasure, copy=True )

        
        #
        # Aspect Beta
        #        
        self.meshQualityFilter.SetTetQualityMeasureToAspectBeta()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['AspectBeta'] = np.array( qualityMeasure, copy=True )

        
        #
        # Aspect Frobenius
        #        
        self.meshQualityFilter.SetTetQualityMeasureToAspectFrobenius()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['AspectFrobenius'] = np.array( qualityMeasure, copy=True )

        
        #
        # Aspect Gamma
        #        
        self.meshQualityFilter.SetTetQualityMeasureToAspectGamma()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['AspectGamma'] = np.array( qualityMeasure, copy=True )

        
        #
        # Aspect Ratio
        #        
        self.meshQualityFilter.SetTetQualityMeasureToAspectRatio()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['AspectRatio'] = np.array( qualityMeasure, copy=True )

        
        #
        # Collapse Ratio
        #        
        self.meshQualityFilter.SetTetQualityMeasureToCollapseRatio()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['CollapseRatio'] = np.array( qualityMeasure, copy=True )
        
        
        #
        # Condition
        #        
        self.meshQualityFilter.SetTetQualityMeasureToCondition()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['Condition'] = np.array( qualityMeasure, copy=True )
        
        
        #
        # Distortion
        #        
        self.meshQualityFilter.SetTetQualityMeasureToDistortion()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['Distortion'] = np.array( qualityMeasure, copy=True )
        
        
        # 
        # Relative Size Squared
        #        
        self.meshQualityFilter.SetTetQualityMeasureToRelativeSizeSquared()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['RelativeSizeSquared'] = np.array( qualityMeasure, copy=True )
        
        
        # 
        # Relative Size Squared
        #        
        self.meshQualityFilter.SetTetQualityMeasureToRelativeSizeSquared()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['RelativeSizeSquared'] = np.array( qualityMeasure, copy=True )
        
        
        # 
        # Shape
        #        
        self.meshQualityFilter.SetTetQualityMeasureToShape()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['Shape'] = np.array( qualityMeasure, copy=True )
        
        
        # 
        # Shape And Size
        #        
        self.meshQualityFilter.SetTetQualityMeasureToShapeAndSize()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['ShapeAndSize'] = np.array( qualityMeasure, copy=True )
        
        
        # 
        # Volume
        #        
        self.meshQualityFilter.SetTetQualityMeasureToVolume()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['Volume'] = np.array( qualityMeasure, copy=True )
        
        
        
        
        
        
        
    
    
    def _calcBasicStatistics( self ):
        ''' Calculate the statistics for each element and  
        '''
        
        xPts = self.nodes[:,0]
        yPts = self.nodes[:,1]
        zPts = self.nodes[:,2]
        
        # big Matrix with coordinate positions. 
        M = np.array( ( xPts[self.elements[:,0]], yPts[self.elements[:,0]], zPts[self.elements[:,0]], 
                        xPts[self.elements[:,1]], yPts[self.elements[:,1]], zPts[self.elements[:,1]],
                        xPts[self.elements[:,2]], yPts[self.elements[:,2]], zPts[self.elements[:,2]],
                        xPts[self.elements[:,3]], yPts[self.elements[:,3]], zPts[self.elements[:,3]] ) ).T
        
        # Vectors within each element...
        a = M[:,0:3] - M[:,3: 6]
        b = M[:,0:3] - M[:,6: 9]
        c = M[:,0:3] - M[:,9:12]
        d = M[:,3:6] - M[:,6: 9]
        e = M[:,3:6] - M[:,9:12]
        f = M[:,6:9] - M[:,9:12]
        
        #
        # Calculate lengths
        #
        self.Lengths = np.array( np.sqrt( ( np.sum(a * a, 1 ), 
                                            np.sum(b * b, 1 ),
                                            np.sum(c * c, 1 ),
                                            np.sum(d * d, 1 ),
                                            np.sum(e * e, 1 ),
                                            np.sum(f * f, 1 ) ) ) ).T
                        
        #plt.hist( self.Lengths[:], bins=120 )
        
        #
        # Calculate volumes:
        # 
        #      | a.(b x c) |
        # V = ---------------
        #            6
         
        self.Vols = np.abs( a[:,0] *(  b[:,1] * c[:,2] - b[:,2] * c[:,1] ) + 
                            a[:,1] *(  b[:,2] * c[:,0] - b[:,0] * c[:,2] ) + 
                            a[:,2] *(  b[:,0] * c[:,1] - b[:,1] * c[:,0] ) ) / 6.0  
        
        print( 'Done.' )
        
        
        
        
    def showMesh( self ):
        
        if not hasattr( self, 'unstructuredGrid' ):
            self._buildVTKMesh()
        
        aTetraMapper = vtk.vtkDataSetMapper()
        aTetraMapper.SetInput( self.unstructuredGrid )
        aTetraActor = vtk.vtkActor()
        aTetraActor.SetMapper( aTetraMapper )
        aTetraActor.AddPosition(4, 0, 0)
        aTetraActor.GetProperty().SetDiffuseColor(0, 1, 0)
            
        ren = vtk.vtkRenderer()
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(ren)
        renWin.SetSize(300, 150)
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        
        ren.SetBackground(.1, .2, .4)
        
        ren.AddActor(aTetraActor)
        ren.ResetCamera()
        ren.GetActiveCamera().Azimuth(30)
        ren.GetActiveCamera().Elevation(20)
        ren.GetActiveCamera().Dolly(2.8)
        ren.ResetCameraClippingRange()
        
        # Render the scene and start interaction.
        iren.Initialize()
        renWin.Render()
        iren.Start()
        
        pass

    
    
    
if __name__ == '__main__' :

    vtkMeshName = 'Q:/philipsBreastProneSupine/referenceState/00/referenceState/surfMeshImpro.1.vtk'
    mesh = vmr.vtkMeshFileReader( vtkMeshName )
    
    stat = meshStatistics( mesh.points, mesh.cells[:,1:] )
    stat.showMesh()




