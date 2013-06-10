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
  DataManagement/mitkDataNodeBoolPropertyFilter.cxx
  DataManagement/mitkDataNodeStringPropertyFilter.cxx
  DataManagement/mitkDataStorageUtils.cxx
  DataManagement/mitkDataStorageListener.cxx
  DataManagement/mitkDataStoragePropertyListener.cxx
  DataManagement/mitkDataStorageVisibilityTracker.cxx
  DataManagement/mitkCoordinateAxesData.cxx
  Rendering/mitkCoordinateAxesVtkMapper3D.cxx
  Rendering/mitkFastPointSetVtkMapper3D.cxx
  Rendering/vtkOpenGLMatrixDrivenCamera.cxx
  DataNodeProperties/mitkAffineTransformParametersDataNodeProperty.cxx
  DataNodeProperties/mitkAffineTransformDataNodeProperty.cxx
  DataNodeProperties/mitkITKRegionParametersDataNodeProperty.cxx
  DataNodeProperties/mitkNamedLookupTableProperty.cxx
  IO/itkAnalyzeImageIO3160.cxx
  IO/itkDRCAnalyzeImageIO3160.cxx
  IO/itkNiftiImageIO3201.cxx
  IO/mitkNifTKItkImageFileReader.cxx
  IO/mitkNifTKItkImageFileIOFactory.cxx
  IO/itkPNMImageIOFactory.cxx
  IO/itkPNMImageIO.cxx
  IO/mitkFileIOUtils.cxx
  Interactions/mitkPointSetUpdate.cxx
)
