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
  Conversion/ImageConversion.cxx
  DataSources/mitkIGITestDataUtils.cxx
  DataSources/mitkIGIDataType.cxx
  DataSources/mitkIGIDataSource.cxx
  DataSources/mitkIGIOpenCVDataType.cxx
  SurfaceReconstruction/SurfaceReconstruction.cxx
  SurfaceReconstruction/SequentialCpuQds.cxx
  SurfaceReconstruction/QDSCommon.cxx
  TrackedImage/mitkTrackedImageCommand.cxx
  TrackedPointer/mitkTrackedPointerCommand.cxx
  PointBasedRegistration/mitkPointBasedRegistration.cxx
  SurfaceBasedRegistration/mitkSurfaceBasedRegistration.cxx
)
