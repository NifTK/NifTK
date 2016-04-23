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
  Common/mitkMIDASEnums.h
)

set(CPP_FILES
  Algorithms/mitkNifTKCoreObjectFactory.cxx
  Algorithms/niftkAffineTransformer.cxx
  Algorithms/mitkNifTKCMC33.cpp
  Algorithms/mitkNifTKImageToSurfaceFilter.cpp
  Algorithms/mitkNifTKMeshSmoother.cpp
  Common/mitkMIDASImageUtils.cxx
  Common/mitkMIDASOrientationUtils.cxx
  Common/mitkPointUtils.cxx
  Common/mitkMergePointClouds.cxx
  DataManagement/mitkDataNodeBoolPropertyFilter.cxx
  DataManagement/mitkDataNodeStringPropertyFilter.cxx
  DataManagement/mitkDataStorageUtils.cxx
  DataManagement/mitkDataStorageListener.cxx
  DataManagement/mitkDataNodePropertyListener.cxx
  DataManagement/mitkDataNodeVisibilityTracker.cxx
  DataManagement/mitkCoordinateAxesData.cxx
  DataManagement/mitkCoordinateAxesDataOpUpdate.cxx
  DataManagement/mitkBasicMesh.cpp
  DataManagement/mitkBasicTriangle.cpp
  DataManagement/mitkBasicVec3D.cpp
  DataManagement/mitkBasicVertex.cpp
  DataManagement/QmitkCmicLogo.cxx
  LookupTables/QmitkLookupTableContainer.cxx
  LookupTables/QmitkLookupTableSaxHandler.cxx
  LookupTables/QmitkLookupTableManager.cxx
  LookupTables/vtkLookupTableUtils.cxx
  Rendering/mitkCoordinateAxesVtkMapper3D.cxx
  Rendering/mitkFastPointSetVtkMapper3D.cxx
  Rendering/niftkCustomVTKAxesActor.cxx
  Rendering/vtkOpenGLMatrixDrivenCamera.cxx
  DataNodeProperties/mitkAffineTransformParametersDataNodeProperty.cxx
  DataNodeProperties/mitkAffineTransformDataNodeProperty.cxx
  DataNodeProperties/mitkITKRegionParametersDataNodeProperty.cxx
  DataNodeProperties/niftkLabeledLookupTableProperty.cxx
  DataNodeProperties/niftkNamedLookupTableProperty.cxx
  IO/mitkFileIOUtils.cxx
  IO/mitkLabelMapReader.cxx
  IO/mitkLabelMapWriter.cxx
  IO/niftkCoreIOMimeTypes.cxx
  Interactions/mitkPointSetUpdate.cxx
  Interactions/niftkAffineTransformDataInteractor3D.cxx
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

