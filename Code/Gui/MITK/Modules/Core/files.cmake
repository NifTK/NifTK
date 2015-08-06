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
  Algorithms/mitkNifTKAffineTransformer.cxx
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
  Rendering/mitkCoordinateAxesVtkMapper3D.cxx
  Rendering/mitkFastPointSetVtkMapper3D.cxx
  Rendering/vtkOpenGLMatrixDrivenCamera.cxx
  DataNodeProperties/mitkAffineTransformParametersDataNodeProperty.cxx
  DataNodeProperties/mitkAffineTransformDataNodeProperty.cxx
  DataNodeProperties/mitkITKRegionParametersDataNodeProperty.cxx
  DataNodeProperties/mitkNamedLookupTableProperty.cxx
  DataNodeProperties/mitkLabeledLookupTableProperty.cxx
  IO/mitkNamedLookupTablePropertySerializer.cxx
  IO/mitkLabeledLookupTablePropertySerializer.cxx
  IO/mitkRawImageReader.cxx
  IO/mitkRawImageWriter.cxx
  IO/mitkRawImageWriterFactory.cxx
  IO/mitkFileIOUtils.cxx
  Interactions/mitkPointSetUpdate.cxx
)

