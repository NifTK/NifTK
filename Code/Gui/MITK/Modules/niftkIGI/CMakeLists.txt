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

MITK_CREATE_MODULE(niftkIGI
  EXPORT_DEFINE NIFTKIGI_EXPORT
  INCLUDE_DIRS DataSources SurfaceReconstruction TrackedImage TrackedPointer PointBasedRegistration SurfaceBasedRegistration
  DEPENDS Mitk MitkExt mitkOpenCVVideoSupport niftkCore mitkCameraCalibration
  PACKAGE_DEPENDS ITK NiftyLink
)

IF(BUILD_TESTING)
  ADD_SUBDIRECTORY(Testing)
ENDIF()  
