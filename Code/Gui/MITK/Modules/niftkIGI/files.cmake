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

set(CPP_FILES
  DataSources/mitkIGITestDataUtils.cxx
  DataSources/mitkIGIDataType.cxx
  DataSources/mitkIGIDataSource.cxx
  DataSources/mitkIGIOpenCVDataType.cxx
  SurfaceReconstruction/SurfaceReconstruction.cxx
  SurfaceReconstruction/SequentialCpuQds.cxx
  SurfaceReconstruction/QDSCommon.cxx
  TrackedImage/mitkTrackedImageCommand.cxx
  TrackedPointer/mitkTrackedPointerManager.cxx
  PointBasedRegistration/mitkPointBasedRegistration.cxx
  PointBasedRegistration/mitkPointsAndNormalsBasedRegistration.cxx
  TagTracking/mitkTagTrackingRegistrationManager.cxx
  SurfaceBasedRegistration/mitkSurfaceBasedRegistration.cxx
)
