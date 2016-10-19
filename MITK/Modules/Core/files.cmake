#/*============================================================================
#
#  NifTK: A software platform for medical image computing.
#
#  Copyright (c) University College London (UCL). All rights reserved.
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.
#
#  See LICENSE.txt in the top level directory for details.
#
#============================================================================*/

set(H_FILES
  Algorithms/niftkCMC33LookUpTable.h
  Common/niftkImageOrientation.h
  Common/niftkImageConversion.h
  DataManagement/niftkLookupTableProviderService.h
)

set(CPP_FILES
  Algorithms/niftkAffineTransformer.cxx
  Algorithms/niftkCMC33.cxx
  Algorithms/niftkImageToSurfaceFilter.cxx
  Algorithms/niftkMeshSmoother.cxx
  Common/niftkCoreObjectFactory.cxx
  Common/niftkFileIOUtils.cxx
  Common/niftkImageUtils.cxx
  Common/niftkImageOrientationUtils.cxx
  Common/niftkPointUtils.cxx
  Common/niftkMergePointClouds.cxx
  Common/niftkBinaryMaskUtils.cxx
  DataManagement/niftkAffineTransformParametersDataNodeProperty.cxx
  DataManagement/niftkAffineTransformDataNodeProperty.cxx
  DataManagement/niftkBasicMesh.cxx
  DataManagement/niftkBasicTriangle.cxx
  DataManagement/niftkBasicVec3D.cxx
  DataManagement/niftkBasicVertex.cxx
  DataManagement/niftkCMICLogo.cxx
  DataManagement/niftkCoordinateAxesData.cxx
  DataManagement/niftkCoordinateAxesDataOpUpdate.cxx
  DataManagement/niftkDataNodeFilter.cxx
  DataManagement/niftkDataNodeBoolPropertyFilter.cxx
  DataManagement/niftkDataNodePropertyListener.cxx
  DataManagement/niftkDataNodeStringPropertyFilter.cxx
  DataManagement/niftkDataNodeVisibilityTracker.cxx
  DataManagement/niftkDataStorageListener.cxx
  DataManagement/niftkDataStorageUtils.cxx
  DataManagement/niftkITKRegionParametersDataNodeProperty.cxx
  DataManagement/niftkLabeledLookupTableProperty.cxx
  DataManagement/niftkLookupTableContainer.cxx
  DataManagement/niftkLookupTableSaxHandler.cxx
  DataManagement/niftkLookupTableManager.cxx
  DataManagement/niftkNamedLookupTableProperty.cxx
  DataManagement/niftkVtkLookupTableUtils.cxx
  Rendering/niftkCoordinateAxesVtkMapper3D.cxx
  Rendering/niftkCustomVTKAxesActor.cxx
  Rendering/niftkFastPointSetVtkMapper3D.cxx
  Rendering/vtkOpenGLMatrixDrivenCamera.cxx
  Interactions/niftkPointSetUpdate.cxx
  Interactions/niftkAffineTransformDataInteractor3D.cxx
  Interactions/niftkInteractionEventObserverMutex.cxx
)

set(MOC_H_FILES
  Interactions/niftkAffineTransformDataInteractor3D.h
)

set(RESOURCE_FILES
  Interactions/AffineTransformConfig.xml
  Interactions/AffineTransformSM.xml
)

set(QRC_FILES
  Resources/niftkCore.qrc
)

