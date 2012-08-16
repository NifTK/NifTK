#! /usr/bin/env python 
# -*- coding: utf-8 -*-


import numpy as np
import vtk
from vtk.util import numpy_support as VN


class meshStatistics:
    ''' Class which builds mesh statistics 
     
    '''
    
    def __init__( self, nodes, elements ):
        
        self.nodes    = nodes
        self.elements = elements
        
        if not isinstance(self.elements, np.ndarray):
            print ( 'Elements must be a numpy array!' )
            return 
        
        if not isinstance(self.nodes, np.ndarray):
            print ( 'Nodes must be a numpy array!' )
            return
        
        self.qualityMeasures = {}
        
        # Calculate statistics for tetrahedra
        if self.elements.shape[1] == 4 :
            self._buildVTKMeshTet()
            self._calcQualityMeasuresTet()

        # Calculate statistics for hexahedra
        if self.elements.shape[1] == 8 :
            self._buildVTKMeshHex()
            self._calcQualityMeasuresHex()
        
        
    
    
    def _buildVTKMeshTet( self ):
        
        
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
        
        
        

    def _buildVTKMeshHex( self ):
        
        self.unstructuredGrid = vtk.vtkUnstructuredGrid()
        pts = vtk.vtkPoints()
        pts.SetData( VN.numpy_to_vtk(self.nodes, deep=True) )
        self.unstructuredGrid.SetPoints( pts )
    
        #
        # generate cells
        #
        for i in range( self.elements.shape[0] ):
            hexa = vtk.vtkHexahedron()
            hexa.GetPointIds().SetId(0, self.elements[ i, 0 ])
            hexa.GetPointIds().SetId(1, self.elements[ i, 1 ])
            hexa.GetPointIds().SetId(2, self.elements[ i, 2 ])    
            hexa.GetPointIds().SetId(3, self.elements[ i, 3 ])
            hexa.GetPointIds().SetId(4, self.elements[ i, 4 ])
            hexa.GetPointIds().SetId(5, self.elements[ i, 5 ])
            hexa.GetPointIds().SetId(6, self.elements[ i, 6 ])
            hexa.GetPointIds().SetId(7, self.elements[ i, 7 ])
            self.unstructuredGrid.InsertNextCell(hexa.GetCellType(), hexa.GetPointIds())
        
        pass
    
        
        
    def _calcQualityMeasuresTet( self ):
        
        
        #
        # iterate through all the different mesh quality measures and store these in the 
        # dictionary self.qualityMeasures
        #
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
        
        
    def _calcQualityMeasuresHex( self ):
        
        #
        # iterate through all the different mesh quality measures and store these in the 
        # dictionary self.qualityMeasures
        #
        self.meshQualityFilter = vtk.vtkMeshQuality()
        self.meshQualityFilter.SetInput( self.unstructuredGrid )
        
        
        #
        # Condition
        #        
        self.meshQualityFilter.SetHexQualityMeasureToCondition()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['Condition'] = np.array( qualityMeasure, copy=True )

        
        #
        # Diagonal
        #        
        self.meshQualityFilter.SetHexQualityMeasureToDiagonal()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['Diagonal'] = np.array( qualityMeasure, copy=True )

        
        #
        # Dimension
        #        
        self.meshQualityFilter.SetHexQualityMeasureToDimension()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['Dimension'] = np.array( qualityMeasure, copy=True )

        
        #
        # Distortion
        #        
        self.meshQualityFilter.SetHexQualityMeasureToDistortion()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['Distortion'] = np.array( qualityMeasure, copy=True )

        
        #
        # EdgeRatio
        #        
        self.meshQualityFilter.SetHexQualityMeasureToEdgeRatio()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['EdgeRatio'] = np.array( qualityMeasure, copy=True )


        #
        # Jacobian
        #        
        self.meshQualityFilter.SetHexQualityMeasureToJacobian()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['Jacobian'] = np.array( qualityMeasure, copy=True )


        #
        # MaxAspectFrobenius
        #        
        self.meshQualityFilter.SetHexQualityMeasureToMaxAspectFrobenius()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['MaxAspectFrobenius'] = np.array( qualityMeasure, copy=True )


        #
        # MaxEdgeRatios
        #        
        self.meshQualityFilter.SetHexQualityMeasureToMaxEdgeRatios()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['MaxEdgeRatios'] = np.array( qualityMeasure, copy=True )


        #
        # MedAspectFrobenius
        #        
        self.meshQualityFilter.SetHexQualityMeasureToMedAspectFrobenius()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['MedAspectFrobenius'] = np.array( qualityMeasure, copy=True )


        #
        # Oddy
        #        
        self.meshQualityFilter.SetHexQualityMeasureToOddy()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['Oddy'] = np.array( qualityMeasure, copy=True )


        #
        # RelativeSizeSquared
        #        
        self.meshQualityFilter.SetHexQualityMeasureToRelativeSizeSquared()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['RelativeSizeSquared'] = np.array( qualityMeasure, copy=True )


        #
        # ScaledJacobian
        #        
        self.meshQualityFilter.SetHexQualityMeasureToScaledJacobian()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['ScaledJacobian'] = np.array( qualityMeasure, copy=True )


        #
        # Shape
        #        
        self.meshQualityFilter.SetHexQualityMeasureToShape()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['Shape'] = np.array( qualityMeasure, copy=True )


        #
        # ShapeAndSize
        #        
        self.meshQualityFilter.SetHexQualityMeasureToShapeAndSize()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['ShapeAndSize'] = np.array( qualityMeasure, copy=True )


        #
        # Shear
        #        
        self.meshQualityFilter.SetHexQualityMeasureToShear()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['Shear'] = np.array( qualityMeasure, copy=True )


        #
        # ShearAndSize
        #        
        self.meshQualityFilter.SetHexQualityMeasureToShearAndSize()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['ShearAndSize'] = np.array( qualityMeasure, copy=True )


        #
        # Skew
        #        
        self.meshQualityFilter.SetHexQualityMeasureToSkew()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['Skew'] = np.array( qualityMeasure, copy=True )


        #
        # Stretch
        #        
        self.meshQualityFilter.SetHexQualityMeasureToStretch()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['Stretch'] = np.array( qualityMeasure, copy=True )


        #
        # Taper
        #        
        self.meshQualityFilter.SetHexQualityMeasureToTaper()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['Taper'] = np.array( qualityMeasure, copy=True )
        
        #
        # Volume
        #        
        self.meshQualityFilter.SetHexQualityMeasureToVolume()
        self.meshQualityFilter.Update()
        
        mqCellData = self.meshQualityFilter.GetOutput().GetCellData()
        qualityMeasure = VN.vtk_to_numpy( mqCellData.GetArray(0) )
        
        self.qualityMeasures['Volume'] = np.array( qualityMeasure, copy=True )

        
        
        
    def showMesh( self ):
        
        #if not hasattr( self, 'unstructuredGrid' ):
        #    self._buildVTKMesh()
        
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
    import nodesAndElementsFromVTKFile as ndsAndEls
    vtkMeshName = 'Q:/philipsBreastProneSupine/referenceState/00/referenceState/surfMeshImpro.1.vtk'
    vtkMeshName = 'W:/philipsBreastProneSupine/referenceState/00/referenceState/surfMesh_VMesh-7_mod.vtk'
    
    N = ndsAndEls.nodesAndElementsFromVTKFile( vtkMeshName )
    
    stat = meshStatistics( N.meshPoints, N.meshCells )
    stat.showMesh()




