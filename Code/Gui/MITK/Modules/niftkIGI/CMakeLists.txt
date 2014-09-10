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

set(_packages NiftyLink)
if(BUILD_PCL)
  list(APPEND _packages PCL)
endif()

MITK_CREATE_MODULE(niftkIGI
  EXPORT_DEFINE NIFTKIGI_EXPORT
  INCLUDE_DIRS CentreLines Common DataSources PointBasedRegistration SurfaceBasedRegistration SurfaceReconstruction TrackedImage TrackedPointer  PCLTest Utils
  DEPENDS MitkCore MitkOpenCVVideoSupport niftkCore MitkCameraCalibration niftkOpenCV
  PACKAGE_DEPENDS ${_packages}
)

# only enable openmp on windows, for now.
# see https://cmicdev.cs.ucl.ac.uk/trac/ticket/2502
if(MSVC)
  if(OPENMP_FOUND)
    set_target_properties(niftkIGI PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
  endif()
endif()

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkIGI)
endif()

if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()
