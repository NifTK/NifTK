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

find_package(OpenMP)

set(_packages ITK NiftyLink)
if(BUILD_PCL)
  list(APPEND _packages PCL)
endif()

MITK_CREATE_MODULE(niftkIGI
  EXPORT_DEFINE NIFTKIGI_EXPORT
  INCLUDE_DIRS DataSources SurfaceReconstruction TrackedImage TrackedPointer PointBasedRegistration SurfaceBasedRegistration PCLTest Utils
  DEPENDS Mitk MitkExt mitkOpenCVVideoSupport niftkCore mitkCameraCalibration niftkOpenCV
  PACKAGE_DEPENDS ${_packages}
)

# only enable openmp on windows, for now.
# see https://cmicdev.cs.ucl.ac.uk/trac/ticket/2502
if(MSVC)
  if(OPENMP_FOUND)
    set_target_properties(niftkIGI PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
  endif()
endif()

if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()  
